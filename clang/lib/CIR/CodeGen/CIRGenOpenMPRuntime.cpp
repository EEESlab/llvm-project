//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementazione del supporto al data sharing (privatizzazione) e alla
// riduzione OpenMP per la codegen CIR.
//
// Questo file contiene le implementazioni di:
//   - RemapGuard: guardia RAII per il ripristino del localDeclMap
//   - OMPDataSharingProcessor: gestione clausole private/firstprivate
//   - OMPReductionProcessor: gestione clausole reduction
//
// Il flusso generale per entrambi i processori è:
//   1. Raccogliere le variabili dalle clausole OpenMP dell'AST Clang
//   2. Creare le op descrittive a livello di modulo (omp.private o
//      omp.declare_reduction)
//   3. Generare i cast fra il sistema di tipi CIR (!cir.ptr<T>) e
//      quello del dialetto OMP (!llvm.ptr)
//   4. Rimappare il localDeclMap affinché il body della regione OpenMP
//      usi le copie private/ridotte anziché le variabili originali
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPRuntime.h"
#include "CIRGenFunction.h"  // Per CIRGenFunction (codegen a livello di funzione)
#include "CIRGenModule.h"    // Per CIRGenModule (codegen a livello di modulo)
#include "clang/AST/Decl.h"  // Per VarDecl (dichiarazioni di variabile)
#include "clang/AST/DeclarationName.h" // Per DeclarationName (nomi degli operatori overloaded)
#include "clang/AST/Expr.h"           // Per DeclRefExpr (riferimenti a variabili nell'AST)
#include "clang/AST/OpenMPClause.h"   // Per OMPPrivateClause, OMPReductionClause, ecc.
#include "clang/CIR/Dialect/IR/CIRDialect.h" // Per i tipi CIR (IntType, SingleType, ecc.)
#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // Per LLVMPointerType, LoadOp, StoreOp, ecc.
#include "mlir/Dialect/OpenMP/OpenMPDialect.h" // Per PrivateClauseOp, DeclareReductionOp
#include "mlir/IR/BuiltinOps.h"  // Per ModuleOp (il modulo MLIR top-level)

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
// RemapGuard — guardia RAII per il ripristino delle mappature
//===----------------------------------------------------------------------===//

// Costruttore: memorizza il CGF e le coppie (variabile, indirizzo originale)
// che dovranno essere ripristinate alla distruzione della guardia.
OMPDataSharingProcessor::RemapGuard::RemapGuard(
    CIRGenFunction &cgf,
    llvm::SmallVector<std::pair<const VarDecl *, Address>> saved)
    : cgf(cgf), savedAddrs(std::move(saved)) {}

// Distruttore: per ogni variabile salvata, ripristina nel localDeclMap
// l'indirizzo originale (quello che puntava alla variabile host,
// non alla copia privata). Questo è fondamentale perché il codice
// emesso dopo il body OpenMP deve tornare a "vedere" le variabili originali.
OMPDataSharingProcessor::RemapGuard::~RemapGuard() {
  for (auto &[vd, addr] : savedAddrs)
    cgf.replaceAddrOfLocalVar(vd, addr);
}

// Move constructor: trasferisce la proprietà delle mappature salvate.
// Necessario perché applyRemapping() restituisce la guardia per valore.
OMPDataSharingProcessor::RemapGuard::RemapGuard(RemapGuard &&other) noexcept
    : cgf(other.cgf), savedAddrs(std::move(other.savedAddrs)) {}

//===----------------------------------------------------------------------===//
// OMPDataSharingProcessor — gestione private / firstprivate
//===----------------------------------------------------------------------===//

// Costruttore: inizializza i riferimenti al CGF, al builder MLIR e alla
// source location. Il vettore entries è inizialmente vuoto.
OMPDataSharingProcessor::OMPDataSharingProcessor(CIRGenFunction &cgf,
                                                 CIRGenBuilderTy &builder,
                                                 mlir::Location loc)
    : cgf(cgf), builder(builder), loc(loc) {}

/// Converte un tipo CIR nel tipo standard MLIR corrispondente.
///
/// Le regioni delle op omp.private e omp.declare_reduction lavorano
/// con tipi standard MLIR (i32, f64, ecc.) e non con tipi CIR
/// (!cir.int<s,32>, !cir.double, ecc.). Questa funzione effettua
/// la conversione necessaria.
///
/// Tipi supportati:
///   - Interi di qualsiasi larghezza (signed/unsigned) → IntegerType
///   - Bool → IntegerType i1
///   - Float/Double/Half/BFloat16/FP80/FP128 → FloatXXType
///   - LongDouble → ricorsione sul tipo sottostante
///   - Pointer → LLVMPointerType (opaque)
mlir::Type
OMPDataSharingProcessor::convertCIRTypeToStdType(mlir::Type cirType) {
  mlir::MLIRContext *ctx = builder.getContext();

  // Tipi interi CIR (es. !cir.int<s, 32>) → tipi interi standard (es. i32).
  // La larghezza viene preservata, ma si perde l'informazione signed/unsigned
  // perché IntegerType di MLIR non la distingue.
  if (auto intTy = mlir::dyn_cast<cir::IntType>(cirType))
    return mlir::IntegerType::get(ctx, intTy.getWidth());

  // Tipo booleano CIR → intero a 1 bit (i1).
  if (mlir::isa<cir::BoolType>(cirType))
    return mlir::IntegerType::get(ctx, 1);

  // Tipi floating point CIR → tipi float standard MLIR.
  // Ogni tipo CIR ha un corrispondente esatto nel sistema di tipi MLIR.
  if (mlir::isa<cir::SingleType>(cirType))    // float (32 bit)
    return mlir::Float32Type::get(ctx);
  if (mlir::isa<cir::DoubleType>(cirType))    // double (64 bit)
    return mlir::Float64Type::get(ctx);
  if (mlir::isa<cir::FP16Type>(cirType))      // half (16 bit IEEE)
    return mlir::Float16Type::get(ctx);
  if (mlir::isa<cir::BF16Type>(cirType))      // bfloat16 (16 bit Google Brain)
    return mlir::BFloat16Type::get(ctx);
  if (mlir::isa<cir::FP80Type>(cirType))      // x87 extended (80 bit)
    return mlir::Float80Type::get(ctx);
  if (mlir::isa<cir::FP128Type>(cirType))     // quad precision (128 bit)
    return mlir::Float128Type::get(ctx);
  // LongDouble è un wrapper attorno a uno dei tipi float sopra.
  // Si converte ricorsivamente il tipo sottostante.
  if (auto ldTy = mlir::dyn_cast<cir::LongDoubleType>(cirType))
    return convertCIRTypeToStdType(ldTy.getUnderlying());

  // Puntatori CIR (!cir.ptr<T>) → puntatori LLVM opachi (!llvm.ptr).
  if (mlir::isa<cir::PointerType>(cirType))
    return mlir::LLVM::LLVMPointerType::get(ctx);

  // Tipo non supportato: emette un errore diagnostico "Not Yet Implemented"
  // anziché far crashare il compilatore con un assert.
  cgf.getCIRGenModule().errorNYI(loc, "private clause for unsupported type");
  return {}; // Ritorna tipo nullo — il chiamante controlla e salta la var
}

/// Crea (o riusa) un'op omp.private a livello di modulo.
///
/// Le op omp.private sono dichiarazioni globali (a livello di ModuleOp)
/// che descrivono come inizializzare e copiare le variabili private.
/// Vengono referenziate per nome simbolico (@nome.privatizer) dalle
/// direttive OpenMP (omp.parallel, omp.wsloop, ecc.).
///
/// Per le variabili 'private': la init region semplicemente yield-a
/// la variabile allocata senza inizializzazione speciale (per scalari).
///
/// Per le variabili 'firstprivate': oltre alla init region, viene
/// creata anche una copy region che fa load/store del valore originale
/// nella copia privata.
void OMPDataSharingProcessor::getOrCreatePrivateOp(
    llvm::StringRef name, mlir::Type stdType,
    mlir::omp::DataSharingClauseType dsType) {
  // Cerca nel modulo se un'op con questo nome esiste già.
  // Se sì, la riusa (per evitare duplicati quando più clausole
  // referenziano la stessa variabile in direttive diverse).
  auto moduleOp = cgf.getCIRGenModule().getModule();
  if (moduleOp.lookupSymbol<mlir::omp::PrivateClauseOp>(name))
    return;

  // Salva e ripristina il punto di inserimento del builder con una
  // InsertionGuard RAII, perché stiamo per spostarci nel body del modulo.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  // Crea l'op omp.private con il nome, il tipo e il tipo di data sharing
  // (Private o FirstPrivate).
  auto privateOp =
      mlir::omp::PrivateClauseOp::create(builder, loc, name, stdType, dsType);

  // Tipo puntatore opaco LLVM, usato come tipo dei block arguments
  // nelle regioni init e copy (il dialetto OMP lavora con !llvm.ptr).
  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());

  // === Init Region ===
  // Inizializza la variabile privata. Per scalari, non serve fare nulla
  // di speciale: il runtime OMP alloca la variabile e noi semplicemente
  // yield-iamo il puntatore all'alloca.
  //
  // La regione ha 2 block arguments di tipo !llvm.ptr:
  //   %arg0 = "mold" — la variabile originale (host), usabile come template
  //                     per l'allocazione (utile per array/struct, non per scalari)
  //   %arg1 = variabile privata già allocata dal runtime
  //
  // Questo pattern replica la funzione initTrivialType() di Flang
  // in PrivateReductionUtils.cpp.
  {
    mlir::Region &initRegion = privateOp.getInitRegion();
    // Crea un blocco con 2 argomenti !llvm.ptr
    mlir::Block *initBlock = builder.createBlock(
        &initRegion, /*insertPt=*/{}, {llvmPtrTy, llvmPtrTy}, {loc, loc});
    builder.setInsertionPointToEnd(initBlock);
    // Yield-a %arg1 (la variabile privata allocata), senza modificarla
    mlir::omp::YieldOp::create(builder, loc,
                                mlir::ValueRange{initBlock->getArgument(1)});
  }

  // === Copy Region (solo per firstprivate) ===
  // Copia il valore dalla variabile originale alla copia privata.
  // Questo implementa la semantica "firstprivate": la copia privata
  // viene inizializzata con il valore che la variabile aveva all'ingresso
  // della regione parallela.
  //
  // La regione ha 2 block arguments di tipo !llvm.ptr:
  //   %arg0 = variabile originale (sorgente, da cui leggere)
  //   %arg1 = variabile privata (destinazione, in cui scrivere)
  //
  // Genera: load da %arg0 → store in %arg1 → yield %arg1
  if (dsType == mlir::omp::DataSharingClauseType::FirstPrivate) {
    mlir::Region &copyRegion = privateOp.getCopyRegion();
    mlir::Block *copyBlock = builder.createBlock(
        &copyRegion, /*insertPt=*/{}, {llvmPtrTy, llvmPtrTy}, {loc, loc});
    builder.setInsertionPointToEnd(copyBlock);
    mlir::Value origPtr = copyBlock->getArgument(0); // Puntatore sorgente
    mlir::Value privPtr = copyBlock->getArgument(1); // Puntatore destinazione
    // Carica il valore dalla variabile originale
    mlir::Value val =
        mlir::LLVM::LoadOp::create(builder, loc, stdType, origPtr);
    // Salva il valore nella copia privata
    mlir::LLVM::StoreOp::create(builder, loc, val, privPtr);
    // Yield-a il puntatore alla copia (ora inizializzata)
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{privPtr});
  }
}

/// Fase 1 del protocollo di data sharing.
///
/// Questa funzione:
/// 1. Itera tutte le clausole OpenMP e filtra quelle di tipo
///    OMPPrivateClause e OMPFirstprivateClause
/// 2. Per ogni variabile in ogni clausola:
///    a. Recupera l'indirizzo corrente dal localDeclMap (l'alloca !cir.ptr<T>)
///    b. Converte il tipo CIR → tipo standard MLIR
///    c. Crea l'op omp.private a livello di modulo
///    d. Salva una entry nei metadati
/// 3. Genera i cast !cir.ptr → !llvm.ptr prima dell'op target e li
///    aggiunge a clauseOps (che verrà poi attaccato all'op OpenMP)
void OMPDataSharingProcessor::processStep1(
    llvm::ArrayRef<const OMPClause *> clauses,
    OMPPrivateClauseOps &clauseOps, mlir::Operation *insertBeforeOp) {

  // Lambda helper che processa una singola variabile.
  // Parametri:
  //   varExpr — l'espressione AST che referenzia la variabile
  //   dsType  — Private o FirstPrivate
  auto processVar = [&](const Expr *varExpr,
                        mlir::omp::DataSharingClauseType dsType) {
    // Dall'espressione AST, ottieni il DeclRefExpr (ignorando parentesi
    // e cast impliciti) e poi la VarDecl sottostante.
    const auto *dre = cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
    const auto *vd = cast<VarDecl>(dre->getDecl());

    // Recupera l'indirizzo corrente della variabile dal localDeclMap.
    // Questo è un Address che contiene sia il puntatore MLIR (!cir.ptr<T>)
    // sia il tipo dell'elemento (T) e l'allineamento.
    Address addr = cgf.getAddrOfLocalVar(vd);
    mlir::Value originalAddr = addr.getPointer();  // Il valore !cir.ptr<T>
    mlir::Type elementType = addr.getElementType(); // Il tipo T

    // Converte il tipo CIR in tipo standard MLIR per l'op omp.private.
    mlir::Type stdType = convertCIRTypeToStdType(elementType);
    if (!stdType)
      return; // Se il tipo non è supportato, errorNYI è già stato emesso

    // Genera il nome simbolico per il privatizer (es. "x.privatizer")
    // e crea/riusa l'op omp.private nel modulo.
    std::string privatizerName = vd->getNameAsString() + ".privatizer";
    getOrCreatePrivateOp(privatizerName, stdType, dsType);

    // Salva i metadati della variabile. Il campo blockArg sarà riempito
    // successivamente da addBlockArgs().
    entries.push_back({vd, originalAddr, elementType, privatizerName, {}});
  };

  // Itera tutte le clausole e processa private e firstprivate.
  for (const OMPClause *c : clauses) {
    if (const auto *privClause = dyn_cast<OMPPrivateClause>(c)) {
      // Clausola private(x, y, ...) — ogni variabile diventa private
      for (const Expr *varExpr : privClause->varlist())
        processVar(varExpr, mlir::omp::DataSharingClauseType::Private);
    } else if (const auto *fpClause = dyn_cast<OMPFirstprivateClause>(c)) {
      // Clausola firstprivate(x, y, ...) — ogni variabile diventa
      // firstprivate (inizializzata con il valore host)
      for (const Expr *varExpr : fpClause->varlist())
        processVar(varExpr, mlir::omp::DataSharingClauseType::FirstPrivate);
    }
  }

  // Genera i cast !cir.ptr<T> → !llvm.ptr per ogni variabile.
  //
  // I cast devono essere inseriti PRIMA dell'op OpenMP target
  // (insertBeforeOp) affinché i valori risultanti siano disponibili
  // (dominino) quando l'op li usa come operandi.
  //
  // Questi valori !llvm.ptr verranno passati all'op OpenMP come
  // attributi private_vars, e i nomi dei privatizer come private_syms.
  if (!entries.empty()) {
    mlir::Type llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());
    // Salva il punto di inserimento corrente e lo sposta prima dell'op target.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertBeforeOp);
    for (auto &entry : entries) {
      // Crea un UnrealizedConversionCastOp: !cir.ptr<T> → !llvm.ptr
      // Questo cast è "unrealized" (non realizzato) perché sarà risolto
      // più tardi durante il lowering CIR → LLVM IR.
      mlir::Value stdPtr =
          mlir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy,
                                                   entry.originalAddr)
              .getResult(0);
      // Aggiunge il puntatore cast-ato e il simbolo del privatizer
      // agli operandi della clausola.
      clauseOps.privateVars.push_back(stdPtr);
      clauseOps.privateSyms.push_back(mlir::FlatSymbolRefAttr::get(
          builder.getContext(), entry.privatizerName));
    }
  }
}

/// Fase 2: aggiunge block arguments alla regione dell'op OpenMP.
///
/// Ogni variabile privata necessita di un block argument di tipo !llvm.ptr
/// nella regione dell'op (es. il body di omp.parallel). Questo block
/// argument riceverà il puntatore alla copia privata dal runtime OpenMP.
void OMPDataSharingProcessor::addBlockArgs(mlir::Block &block) {
  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto &entry : entries)
    // Aggiunge un argomento !llvm.ptr al blocco e salva il riferimento
    // nella entry per l'uso successivo in applyRemapping().
    entry.blockArg = block.addArgument(llvmPtrTy, loc);
}

/// Fase 3: rimappa le variabili locali alle copie private.
///
/// Per ogni variabile privata:
/// 1. Crea un cast inverso !llvm.ptr → !cir.ptr<T> dal block argument
///    (che è !llvm.ptr) al tipo CIR originale della variabile
/// 2. Salva la mappatura corrente (variabile → indirizzo originale)
/// 3. Sostituisce la mappatura con il nuovo indirizzo (la copia privata)
///
/// Restituisce una RemapGuard che ripristinerà le mappature originali
/// quando esce dallo scope, permettendo al codice successivo al body
/// OpenMP di tornare a usare le variabili host.
OMPDataSharingProcessor::RemapGuard
OMPDataSharingProcessor::applyRemapping() {
  llvm::SmallVector<std::pair<const VarDecl *, Address>> saved;
  for (auto &entry : entries) {
    // Cast !llvm.ptr → !cir.ptr<T>: converte il block argument (opaco)
    // nel tipo di puntatore CIR originale della variabile.
    mlir::Value cirPtr =
        mlir::UnrealizedConversionCastOp::create(
            builder, loc, entry.originalAddr.getType(), entry.blockArg)
            .getResult(0);
    // Salva l'indirizzo attuale per il ripristino successivo.
    saved.push_back({entry.varDecl, cgf.getAddrOfLocalVar(entry.varDecl)});
    // Rimappa: ora quando il codice nel body farà cgf.getAddrOfLocalVar(vd),
    // otterrà il cirPtr (la copia privata) anziché l'originale.
    // L'allineamento CharUnits::One() è conservativo.
    cgf.replaceAddrOfLocalVar(
        entry.varDecl,
        Address(cirPtr, entry.elementType, CharUnits::One()));
  }
  // Restituisce la guardia che ripristinerà tutto alla distruzione.
  return RemapGuard(cgf, std::move(saved));
}

//===----------------------------------------------------------------------===//
// OMPReductionProcessor — gestione della clausola reduction
//===----------------------------------------------------------------------===//

// Costruttore: identico al DataSharingProcessor.
OMPReductionProcessor::OMPReductionProcessor(CIRGenFunction &cgf,
                                             CIRGenBuilderTy &builder,
                                             mlir::Location loc)
    : cgf(cgf), builder(builder), loc(loc) {}

/// Conversione tipi CIR → standard MLIR per il contesto reduction.
/// Stessa logica della versione in OMPDataSharingProcessor, ma senza
/// il supporto ai puntatori (non si fa riduzione su puntatori).
mlir::Type
OMPReductionProcessor::convertCIRTypeToStdType(mlir::Type cirType) {
  mlir::MLIRContext *ctx = builder.getContext();

  if (auto intTy = mlir::dyn_cast<cir::IntType>(cirType))
    return mlir::IntegerType::get(ctx, intTy.getWidth());
  if (mlir::isa<cir::BoolType>(cirType))
    return mlir::IntegerType::get(ctx, 1);
  if (mlir::isa<cir::SingleType>(cirType))
    return mlir::Float32Type::get(ctx);
  if (mlir::isa<cir::DoubleType>(cirType))
    return mlir::Float64Type::get(ctx);
  if (mlir::isa<cir::FP16Type>(cirType))
    return mlir::Float16Type::get(ctx);
  if (mlir::isa<cir::BF16Type>(cirType))
    return mlir::BFloat16Type::get(ctx);
  if (mlir::isa<cir::FP80Type>(cirType))
    return mlir::Float80Type::get(ctx);
  if (mlir::isa<cir::FP128Type>(cirType))
    return mlir::Float128Type::get(ctx);
  if (auto ldTy = mlir::dyn_cast<cir::LongDoubleType>(cirType))
    return convertCIRTypeToStdType(ldTy.getUnderlying());

  cgf.getCIRGenModule().errorNYI(loc,
                                 "reduction clause for unsupported type");
  return {};
}

/// Restituisce l'elemento neutro (identità) per la riduzione.
///
/// L'elemento neutro è il valore iniziale della variabile di riduzione
/// per ogni thread. È scelto in modo che combinato con qualsiasi
/// valore x tramite l'operatore di riduzione, restituisca x:
///   - Add/Or/Xor:  0 (perché x + 0 = x, x | 0 = x, x ^ 0 = x)
///   - Mul/And:      1 (perché x * 1 = x, x & all_ones = x)
///   - LogicalAnd:   1 (true, perché x && true = x)
///   - LogicalOr:    0 (false, perché x || false = x)
///
/// Per i float, solo Add (0.0) e Mul (1.0) sono supportati.
mlir::Value OMPReductionProcessor::getReductionInitValue(
    mlir::Type stdType, OMPReductionKind redKind) {
  // === Caso intero ===
  if (mlir::isa<mlir::IntegerType>(stdType)) {
    int64_t initVal = 0;
    switch (redKind) {
    case OMPReductionKind::Add:        // Elemento neutro della somma: 0
    case OMPReductionKind::BitwiseOr:  // Elemento neutro dell'OR bit a bit: 0
    case OMPReductionKind::BitwiseXor: // Elemento neutro dello XOR: 0
    case OMPReductionKind::LogicalOr:  // Elemento neutro dell'OR logico: false (0)
      initVal = 0;
      break;
    case OMPReductionKind::Multiply:   // Elemento neutro del prodotto: 1
    case OMPReductionKind::BitwiseAnd: // Elemento neutro dell'AND bit a bit: all-ones
                                       // (Nota: 1 per i1; per larghezze maggiori
                                       //  dovrebbe essere ~0, ma per ora usa 1)
    case OMPReductionKind::LogicalAnd: // Elemento neutro dell'AND logico: true (1)
      initVal = 1;
      break;
    }
    // Crea una costante LLVM IR del tipo e valore appropriati.
    return mlir::LLVM::ConstantOp::create(
        builder, loc, stdType,
        builder.getIntegerAttr(stdType, initVal));
  }

  // === Caso floating point ===
  if (mlir::isa<mlir::FloatType>(stdType)) {
    double initVal = 0.0;
    switch (redKind) {
    case OMPReductionKind::Add:      // Elemento neutro della somma float: 0.0
      initVal = 0.0;
      break;
    case OMPReductionKind::Multiply: // Elemento neutro del prodotto float: 1.0
      initVal = 1.0;
      break;
    default:
      // Operatori bit a bit e logici non sono supportati per i float
      // (non hanno senso semantico).
      cgf.getCIRGenModule().errorNYI(
          loc, "reduction init value for non-arithmetic float operator");
      return {};
    }
    return mlir::LLVM::ConstantOp::create(
        builder, loc, stdType,
        builder.getFloatAttr(stdType, initVal));
  }

  // Tipo non supportato (es. struct, array).
  cgf.getCIRGenModule().errorNYI(loc, "reduction init for unsupported type");
  return {};
}

/// Crea l'operazione di combinazione (combiner) per la riduzione.
///
/// Il combiner è l'operazione binaria che combina i risultati parziali
/// di due thread. Ad esempio, per reduction(+:x), il combiner è
/// una somma: result = lhs + rhs.
///
/// Per ogni operatore, ci sono varianti intere e float separate
/// (es. AddOp vs FAddOp) perché LLVM IR distingue aritmetica
/// intera da floating point.
mlir::Value OMPReductionProcessor::createCombiner(mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  mlir::Type stdType,
                                                  OMPReductionKind redKind) {
  bool isInt = mlir::isa<mlir::IntegerType>(stdType);
  bool isFloat = mlir::isa<mlir::FloatType>(stdType);

  switch (redKind) {
  case OMPReductionKind::Add:
    // Somma: add per interi, fadd per float
    if (isInt)
      return mlir::LLVM::AddOp::create(builder, loc, lhs, rhs);
    if (isFloat)
      return mlir::LLVM::FAddOp::create(builder, loc, lhs, rhs);
    break;
  case OMPReductionKind::Multiply:
    // Prodotto: mul per interi, fmul per float
    if (isInt)
      return mlir::LLVM::MulOp::create(builder, loc, lhs, rhs);
    if (isFloat)
      return mlir::LLVM::FMulOp::create(builder, loc, lhs, rhs);
    break;
  case OMPReductionKind::BitwiseAnd:
    // AND bit a bit: solo per interi
    assert(isInt && "bitwise AND requires integer type");
    return mlir::LLVM::AndOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::BitwiseOr:
    // OR bit a bit: solo per interi
    assert(isInt && "bitwise OR requires integer type");
    return mlir::LLVM::OrOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::BitwiseXor:
    // XOR bit a bit: solo per interi
    assert(isInt && "bitwise XOR requires integer type");
    return mlir::LLVM::XOrOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::LogicalAnd:
    // AND logico: implementato come AND bit a bit su interi
    // (funziona perché i valori booleani sono 0 o 1)
    assert(isInt && "logical AND requires integer type");
    return mlir::LLVM::AndOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::LogicalOr:
    // OR logico: implementato come OR bit a bit su interi
    assert(isInt && "logical OR requires integer type");
    return mlir::LLVM::OrOp::create(builder, loc, lhs, rhs);
  }

  // Combinazione tipo/operatore non supportata.
  cgf.getCIRGenModule().errorNYI(loc, "reduction combiner for type/op combo");
  return {};
}

/// Crea (o riusa) un'op omp.declare_reduction a livello di modulo.
///
/// A differenza di omp.private, omp.declare_reduction ha due regioni:
///   1. Initializer: produce l'elemento neutro (es. 0 per la somma)
///   2. Combiner: descrive come combinare due valori parziali
///
/// Queste regioni lavorano con valori scalari (by-value), non con
/// puntatori. Il runtime OpenMP gestisce l'allocazione thread-local
/// e il load/store; le regioni descrivono solo la logica di riduzione.
void OMPReductionProcessor::getOrCreateDeclareReduction(
    llvm::StringRef name, mlir::Type stdType, OMPReductionKind redKind) {
  // Controlla se l'op esiste già nel modulo (evita duplicati).
  auto moduleOp = cgf.getCIRGenModule().getModule();
  if (moduleOp.lookupSymbol<mlir::omp::DeclareReductionOp>(name))
    return;

  // Sposta il punto di inserimento all'inizio del modulo.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  // Crea l'op omp.declare_reduction. Il campo byref_element_type è vuoto
  // perché usiamo riduzione by-value per scalari.
  auto declOp = mlir::omp::DeclareReductionOp::create(
      builder, loc, name, stdType, /*byref_element_type=*/{});

  // === Initializer Region ===
  // Ha 1 block argument del tipo scalare (stdType).
  // Produce l'elemento neutro della riduzione (identità dell'operatore).
  {
    mlir::Region &initRegion = declOp.getInitializerRegion();
    mlir::Block *initBlock =
        builder.createBlock(&initRegion, initRegion.end(), {stdType}, {loc});
    builder.setInsertionPointToEnd(initBlock);
    // Genera la costante dell'elemento neutro (0, 1, 0.0, 1.0, ecc.)
    mlir::Value initVal = getReductionInitValue(stdType, redKind);
    // Yield-a il valore iniziale
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{initVal});
  }

  // === Combiner Region ===
  // Ha 2 block arguments del tipo scalare: il risultato parziale del
  // thread corrente e quello di un altro thread.
  // Produce il risultato combinato (es. lhs + rhs per la somma).
  {
    mlir::Region &combinerRegion = declOp.getReductionRegion();
    mlir::Block *combBlock = builder.createBlock(
        &combinerRegion, combinerRegion.end(), {stdType, stdType}, {loc, loc});
    builder.setInsertionPointToEnd(combBlock);
    // Crea l'operazione di combinazione (add, mul, and, or, xor)
    mlir::Value combined = createCombiner(
        combBlock->getArgument(0),  // Risultato parziale thread A
        combBlock->getArgument(1),  // Risultato parziale thread B
        stdType, redKind);
    // Yield-a il risultato combinato
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{combined});
  }
}

/// Mappa un operatore overloaded C++ al corrispondente OMPReductionKind.
///
/// Questa funzione traduce l'enumerazione OverloadedOperatorKind di Clang
/// (OO_Plus, OO_Star, ecc.) nella nostra enumerazione OMPReductionKind.
///
/// Nota: OO_Minus (il segno -) è mappato a Add perché in OpenMP
/// reduction(-:x) usa la somma come combiner. La sottrazione non è
/// associativa, ma la specifica OpenMP definisce che reduction(-:x)
/// si comporta come reduction(+:x) con elemento neutro 0.
static std::optional<OMPReductionKind>
mapOverloadedOpToReductionKind(OverloadedOperatorKind op) {
  switch (op) {
  case OO_Plus:
  case OO_Minus: // reduction(-:x) usa lo stesso combiner di +
    return OMPReductionKind::Add;
  case OO_Star:     // reduction(*:x) → moltiplicazione
    return OMPReductionKind::Multiply;
  case OO_Amp:      // reduction(&:x) → AND bit a bit
    return OMPReductionKind::BitwiseAnd;
  case OO_Pipe:     // reduction(|:x) → OR bit a bit
    return OMPReductionKind::BitwiseOr;
  case OO_Caret:    // reduction(^:x) → XOR bit a bit
    return OMPReductionKind::BitwiseXor;
  case OO_AmpAmp:   // reduction(&&:x) → AND logico
    return OMPReductionKind::LogicalAnd;
  case OO_PipePipe: // reduction(||:x) → OR logico
    return OMPReductionKind::LogicalOr;
  default:
    // Operatore non supportato (es. max, min, operatori user-defined)
    return std::nullopt;
  }
}

/// Restituisce un nome leggibile per un OMPReductionKind.
/// Usato per costruire il nome simbolico dell'op omp.declare_reduction
/// (es. "add_x", "multiply_result").
static llvm::StringRef getReductionKindName(OMPReductionKind kind) {
  switch (kind) {
  case OMPReductionKind::Add:
    return "add";
  case OMPReductionKind::Multiply:
    return "multiply";
  case OMPReductionKind::BitwiseAnd:
    return "band";
  case OMPReductionKind::BitwiseOr:
    return "bor";
  case OMPReductionKind::BitwiseXor:
    return "bxor";
  case OMPReductionKind::LogicalAnd:
    return "land";
  case OMPReductionKind::LogicalOr:
    return "lor";
  }
  llvm_unreachable("unknown reduction kind");
}

/// Fase 1 del processore di riduzione.
///
/// Itera tutte le clausole OpenMP e per ogni clausola reduction:
/// 1. Determina il tipo di operatore (add, mul, and, ecc.) dall'AST
/// 2. Per ogni variabile nella clausola:
///    a. Recupera l'indirizzo e il tipo dal localDeclMap
///    b. Crea l'op omp.declare_reduction a livello di modulo
///    c. Genera il cast !cir.ptr → !llvm.ptr
///    d. Aggiunge gli operandi a clauseOps
///
/// Il nome dell'op è formato da: kindName + "_" + varName
/// (es. "add_sum", "multiply_product").
void OMPReductionProcessor::processReductionVars(
    llvm::ArrayRef<const OMPClause *> clauses,
    OMPReductionClauseOps &clauseOps, mlir::Operation *insertBeforeOp) {

  for (const OMPClause *c : clauses) {
    // Filtra solo le clausole di tipo reduction. Le altre (private,
    // firstprivate, schedule, ecc.) vengono ignorate.
    const auto *redClause = dyn_cast<OMPReductionClause>(c);
    if (!redClause)
      continue;

    // Determina l'operatore di riduzione dalla clausola AST.
    // La clausola reduction(+:x) ha come "name" l'operatore +,
    // rappresentato internamente come OverloadedOperatorKind OO_Plus.
    DeclarationName redName = redClause->getNameInfo().getName();
    OverloadedOperatorKind ooKind = redName.getCXXOverloadedOperator();

    // Mappa l'operatore Clang al nostro OMPReductionKind.
    auto redKind = mapOverloadedOpToReductionKind(ooKind);
    if (!redKind) {
      // Operatore non supportato (es. max, min). Emette errore e continua.
      cgf.getCIRGenModule().errorNYI(
          redClause->getBeginLoc(),
          "reduction clause with unsupported operator");
      continue;
    }

    // Processa ogni variabile nella lista della clausola reduction.
    for (const Expr *varExpr : redClause->varlist()) {
      // Estrai la VarDecl dall'espressione AST.
      const auto *dre = cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
      const auto *vd = cast<VarDecl>(dre->getDecl());

      // Recupera l'indirizzo e il tipo dal localDeclMap.
      Address addr = cgf.getAddrOfLocalVar(vd);
      mlir::Value originalAddr = addr.getPointer();
      mlir::Type elementType = addr.getElementType();

      // Converte il tipo CIR in tipo standard MLIR.
      mlir::Type stdType = convertCIRTypeToStdType(elementType);
      if (!stdType)
        continue; // Tipo non supportato, errore già emesso

      // Costruisce il nome simbolico (es. "add_sum") e crea l'op
      // omp.declare_reduction nel modulo.
      std::string declName =
          (getReductionKindName(*redKind) + "_" + vd->getNameAsString())
              .str();
      getOrCreateDeclareReduction(declName, stdType, *redKind);

      // Salva i metadati della variabile.
      entries.push_back({vd, originalAddr, elementType, declName, {}});
    }
  }

  // Genera i cast !cir.ptr → !llvm.ptr e popola clauseOps.
  // Stessa logica del DataSharingProcessor, ma con campi diversi
  // (reductionVars/reductionSyms/reductionByref invece di
  // privateVars/privateSyms).
  if (!entries.empty()) {
    mlir::Type llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertBeforeOp);
    for (auto &entry : entries) {
      // Cast !cir.ptr<T> → !llvm.ptr
      mlir::Value stdPtr =
          mlir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy,
                                                   entry.originalAddr)
              .getResult(0);
      clauseOps.reductionVars.push_back(stdPtr);
      // Riferimento simbolico all'op omp.declare_reduction
      clauseOps.reductionSyms.push_back(mlir::FlatSymbolRefAttr::get(
          builder.getContext(), entry.reductionName));
      // By-reference = false: per scalari, la riduzione è by-value.
      // Il runtime gestisce allocazione e load/store thread-local.
      clauseOps.reductionByref.push_back(false);
    }
  }
}

/// Fase 2: aggiunge block arguments per le variabili di riduzione.
/// Identico al DataSharingProcessor::addBlockArgs.
void OMPReductionProcessor::addBlockArgs(mlir::Block &block) {
  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto &entry : entries)
    entry.blockArg = block.addArgument(llvmPtrTy, loc);
}

/// Fase 3: rimappa localDeclMap per le variabili di riduzione.
///
/// Identico al DataSharingProcessor::applyRemapping():
/// - Crea cast !llvm.ptr → !cir.ptr<T> per ogni block argument
/// - Salva la mappatura originale
/// - Rimappa la variabile alla copia thread-local
/// - Restituisce RemapGuard per il ripristino automatico
///
/// Riusa la stessa classe RemapGuard del DataSharingProcessor perché
/// la logica di salvataggio/ripristino è identica.
OMPDataSharingProcessor::RemapGuard
OMPReductionProcessor::applyRemapping() {
  llvm::SmallVector<std::pair<const VarDecl *, Address>> saved;
  for (auto &entry : entries) {
    // Cast inverso: !llvm.ptr → !cir.ptr<T>
    mlir::Value cirPtr =
        mlir::UnrealizedConversionCastOp::create(
            builder, loc, entry.originalAddr.getType(), entry.blockArg)
            .getResult(0);
    // Salva mappatura originale
    saved.push_back({entry.varDecl, cgf.getAddrOfLocalVar(entry.varDecl)});
    // Rimappa alla copia thread-local
    cgf.replaceAddrOfLocalVar(
        entry.varDecl,
        Address(cirPtr, entry.elementType, CharUnits::One()));
  }
  return OMPDataSharingProcessor::RemapGuard(cgf, std::move(saved));
}
