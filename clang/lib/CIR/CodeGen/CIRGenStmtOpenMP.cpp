//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emissione delle direttive OpenMP (statement) come codice CIR.
//
// Questo file contiene gli emitter per le direttive OpenMP supportate:
//   - parallel:      emitOMPParallelDirective
//   - for (wsloop):  emitOMPForDirective
//   - parallel for:  emitOMPParallelForDirective
//   - single:        emitOMPSingleDirective
//   - master:        emitOMPMasterDirective
//   - barrier:       emitOMPBarrierDirective
//   - task:          emitOMPTaskDirective
//   - taskwait:      emitOMPTaskwaitDirective
//
// Le direttive non ancora implementate emettono un errore "Not Yet
// Implemented" (NYI) tramite errorNYI.
//
// Il flusso generale per ogni direttiva implementata è:
//   1. Creare l'op MLIR del dialetto OMP (es. omp.parallel)
//   2. Processare le clausole (emitOpenMPClauses)
//   3. Gestire data sharing (OMPDataSharingProcessor) e riduzione
//      (OMPReductionProcessor) se la direttiva lo richiede
//   4. Creare la regione con block arguments e remapping
//   5. Emettere il body della direttiva
//   6. Terminare la regione con omp.terminator o omp.yield
//
// Per i loop (omp for), il flusso è più complesso:
//   - extractOMPLoopBounds estrae i limiti dal ForStmt C/C++
//   - emitOMPForDirective crea l'op omp.wsloop
//   - emitForStmt (in CIRGenStmt.cpp) crea l'op omp.loop_nest
//   - Il remapping private/reduction avviene nel body del loop_nest
//     (non nel wsloop) per rispettare il vincolo "exactly one nested op"
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/StmtOpenMP.h"           // Per OMPParallelDirective, OMPForDirective, ecc.
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h" // Per OMPD_parallel e simili

using namespace clang;
using namespace clang::CIRGen;

// =====================================================================
// Stub NYI (direttive non ancora implementate)
// =====================================================================

mlir::LogicalResult
CIRGenFunction::emitOMPScopeDirective(const OMPScopeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScopeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPErrorDirective(const OMPErrorDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPErrorDirective");
  return mlir::failure();
}

// =====================================================================
// emitOMPParallelDirective — #pragma omp parallel
// =====================================================================

/// Emette una direttiva `#pragma omp parallel { body }`.
///
/// Genera un'op `omp.parallel` con:
///   - Clausole non-private (proc_bind, ecc.) tramite emitOpenMPClauses
///   - Variabili private/firstprivate tramite OMPDataSharingProcessor
///   - Variabili di riduzione tramite OMPReductionProcessor
///   - Un body con remapping RAII delle variabili
///
/// Struttura IR risultante:
///   omp.parallel private(@x.privatizer %x) reduction(@add_sum %sum) {
///     // block args: %priv_x (!llvm.ptr), %red_sum (!llvm.ptr)
///     // cast !llvm.ptr → !cir.ptr<T>
///     // ... body emesso con variabili rimappate ...
///     omp.terminator
///   }
mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::LogicalResult res = mlir::success();
  llvm::SmallVector<mlir::Type> retTy;     // Tipo di ritorno (vuoto per parallel)
  llvm::SmallVector<mlir::Value> operands; // Operandi iniziali (vuoti, aggiunti dopo)
  mlir::Location begin = getLoc(s.getBeginLoc()); // Source location di inizio
  mlir::Location end = getLoc(s.getEndLoc());     // Source location di fine

  // Crea l'op omp.parallel. Gli operandi (private_vars, reduction_vars)
  // verranno aggiunti subito dopo.
  auto parallelOp =
      mlir::omp::ParallelOp::create(builder, begin, retTy, operands);

  // Processa le clausole non gestite dai processori specializzati
  // (es. proc_bind). Le clausole private/reduction sono no-op qui.
  emitOpenMPClauses(parallelOp, s.clauses());

  // === Data Sharing: gestione private/firstprivate ===
  // Crea il processore, raccoglie le variabili dalle clausole,
  // genera le op omp.private a livello di modulo,
  // e produce i cast !cir.ptr → !llvm.ptr come operandi.
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, parallelOp);

  // Se ci sono variabili private, attacca gli operandi all'op parallel:
  //   - private_vars: lista di puntatori !llvm.ptr alle variabili originali
  //   - private_syms: lista di simboli (@nome.privatizer) che referenziano
  //     le op omp.private a livello di modulo
  if (dsp.hasPrivateVars()) {
    parallelOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    parallelOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // === Riduzione: gestione reduction ===
  // Stessa logica della privatizzazione, ma crea op omp.declare_reduction.
  OMPReductionClauseOps redClauseOps;
  OMPReductionProcessor rdp(*this, builder, begin);
  rdp.processReductionVars(s.clauses(), redClauseOps, parallelOp);

  // Se ci sono variabili di riduzione, attacca gli operandi:
  //   - reduction_vars: lista di puntatori !llvm.ptr
  //   - reduction_syms: lista di simboli (@add_sum, ecc.)
  //   - reduction_byref: array di bool (false per scalari = by-value)
  if (rdp.hasReductionVars()) {
    parallelOp.getReductionVarsMutable().append(redClauseOps.reductionVars);
    parallelOp.setReductionSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), redClauseOps.reductionSyms));
    parallelOp.setReductionByrefAttr(
        mlir::DenseBoolArrayAttr::get(builder.getContext(),
                                      redClauseOps.reductionByref));
  }

  // === Creazione della regione e emissione del body ===
  {
    // Crea un blocco vuoto nella regione dell'op parallel.
    mlir::Block &block = parallelOp.getRegion().emplaceBlock();

    // Aggiunge block arguments !llvm.ptr per le variabili private e ridotte.
    // Questi argomenti riceveranno i puntatori alle copie thread-local
    // dal runtime OpenMP.
    dsp.addBlockArgs(block);
    rdp.addBlockArgs(block);

    // Salva e ripristina il punto di inserimento del builder.
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    // Rimappa le variabili nel localDeclMap: i block arguments (!llvm.ptr)
    // vengono castati a !cir.ptr<T> e sostituiti alle variabili originali.
    // Le guardie RAII ripristinano le mappature originali quando escono
    // dallo scope (alla fine del blocco {}).
    auto remapGuard = dsp.applyRemapping();
    auto redRemapGuard = rdp.applyRemapping();

    // Crea uno scope lessicale per il body della regione parallela.
    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    // Funzionalità non ancora supportate.
    if (s.hasCancel())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Cancel");
    if (s.getTaskReductionRefExpr())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Task Reduction");

    // Recupera il body della direttiva parallel.
    // Le direttive OpenMP wrappano il body in un CapturedStmt per
    // il supporto all'outlining (anche se CIR non fa outlining qui).
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
    const Stmt *bodyStmt = cs->getCapturedStmt();

    // Emette il body della regione parallela. useCurrentScope=true
    // perché il LexicalScope è già stato creato sopra.
    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    // Termina la regione con omp.terminator (obbligatorio per omp.parallel).
    // Il terminator segnala al runtime OpenMP la fine della regione parallela.
    mlir::omp::TerminatorOp::create(builder, end);
  }
  // Qui le guardie RAII (remapGuard, redRemapGuard) vengono distrutte,
  // ripristinando le mappature originali nel localDeclMap.

  return res;
}

// =====================================================================
// Helper per l'estrazione dei limiti dei loop OpenMP
// =====================================================================

namespace {

/// Estrae un valore intero letterale da un'espressione, se presente.
/// Usato per ottimizzare i casi comuni come `i = 0` o `i < 100`
/// dove il bound è una costante nota a compile-time.
static std::optional<int64_t> getIntLiteralValue(const Expr *expr) {
  if (const auto *intLit = dyn_cast<IntegerLiteral>(expr->IgnoreImpCasts()))
    return intLit->getValue().getSExtValue();
  return std::nullopt;
}

/// Assicura che un valore CIR abbia il tipo intero CIR specificato.
///
/// Se il valore è un puntatore, prima lo carica (load).
/// Se il tipo non corrisponde, inserisce un cast integrale.
/// Questo è necessario perché i limiti del loop devono avere tutti
/// lo stesso tipo (quello della variabile di induzione).
static mlir::Value ensureCIRIntType(CIRGenBuilderTy &builder,
                                    mlir::Location loc, mlir::Value cirValue,
                                    cir::IntType targetCIRType) {
  // Se il valore è un puntatore, caricalo prima
  if (mlir::isa<cir::PointerType>(cirValue.getType()))
    cirValue = cir::LoadOp::create(builder, loc, cirValue).getResult();

  // Se il tipo è già quello desiderato, restituisci direttamente
  if (cirValue.getType() == targetCIRType)
    return cirValue;

  // Altrimenti, inserisci un cast integrale (es. i64 → i32)
  return builder.createCast(loc, cir::CastKind::integral, cirValue,
                            targetCIRType);
}

/// Converte un valore intero CIR in un intero standard MLIR.
///
/// L'op omp.loop_nest richiede operandi di tipo IntLikeType
/// (AnyInteger | Index), non tipi CIR. Questa funzione inserisce
/// un UnrealizedConversionCastOp per il bridge tra i due sistemi.
static mlir::Value cirIntToStdInt(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value cirValue) {
  auto cirIntType = mlir::cast<cir::IntType>(cirValue.getType());
  // Crea il tipo intero standard con la stessa larghezza
  mlir::Type stdIntType = builder.getIntegerType(cirIntType.getWidth());
  // Cast CIR int → standard int (sarà risolto durante il lowering)
  return mlir::UnrealizedConversionCastOp::create(builder, loc, stdIntType,
                                                  cirValue)
      .getResult(0);
}
} // anonymous namespace

/// Estrae i limiti di un singolo ForStmt C/C++ per l'uso in omp.loop_nest.
///
/// Analizza le 3 parti del for-loop (init/cond/inc) e produce:
///   - lowerBound: valore iniziale della variabile di induzione
///   - upperBound: limite superiore del loop
///   - step: passo di incremento
///   - inclusive: se il confronto è <= (true) o < (false)
///
/// Forme supportate per init:
///   - `int i = 0`   (DeclStmt — variabile dichiarata nel for)
///   - `i = 0`       (BinaryOperator — variabile dichiarata fuori dal for)
///
/// Forme supportate per cond:
///   - `i < N`, `i <= N` (variabile a sinistra, bound a destra)
///   - `N > i`, `N >= i` (bound a sinistra, variabile a destra)
///
/// Forme supportate per inc:
///   - `i++`, `i--`         (UnaryOperator)
///   - `i += step`          (CompoundAssignment)
///   - `i = i + step`       (Assignment con addizione)
///   - `i = step + i`       (Assignment con addizione commutata)
static mlir::LogicalResult extractSingleLoopBounds(
    CIRGenFunction &cgf, CIRGenBuilderTy &builder,
    const ForStmt *forStmt, mlir::Location loc,
    CIRGenFunction::LoopBounds &bounds) {

  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false;       // true se il confronto è <=, false se <
  Address savedAddr = Address::invalid(); // Indirizzo originale per IV dichiarate fuori dal loop

  // === Estrazione della variabile di induzione e del lower bound ===

  const VarDecl *varDecl = nullptr;
  const Expr *initExpr = nullptr;

  // Caso 1: `for (int i = 0; ...)` — la variabile è dichiarata nell'init.
  // Il DeclStmt contiene sia la dichiarazione che l'inizializzazione.
  if (const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit())) {
    varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl());
    if (!varDecl || !varDecl->hasInit())
      return mlir::failure();
    initExpr = varDecl->getInit();
  }
  // Caso 2: `for (i = 0; ...)` — la variabile è dichiarata altrove.
  // L'init è un'assegnazione (BinaryOperator con =).
  else if (const auto *binOp = dyn_cast_or_null<BinaryOperator>(
                 forStmt->getInit())) {
    if (!binOp->isAssignmentOp())
      return mlir::failure();
    const auto *declRef =
        dyn_cast<DeclRefExpr>(binOp->getLHS()->IgnoreParenImpCasts());
    if (!declRef)
      return mlir::failure();
    varDecl = dyn_cast<VarDecl>(declRef->getDecl());
    initExpr = binOp->getRHS();
  }

  if (!varDecl || !initExpr)
    return mlir::failure();

  // Determina il tipo intero CIR della variabile di induzione.
  // Tutti i limiti (lower, upper, step) verranno convertiti a questo tipo.
  QualType loopVarQType = varDecl->getType();
  auto cirType = cgf.convertType(loopVarQType);
  auto cirIntType = mlir::cast<cir::IntType>(cirType);

  // === Estrazione del lower bound ===
  // Se è un letterale intero, crea direttamente una costante CIR.
  // Altrimenti, emette il codice per valutare l'espressione.
  if (auto constVal = getIntLiteralValue(initExpr)) {
    lowerBound = builder.getConstInt(loc, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = cgf.emitScalarExpr(initExpr);
    lowerBound = ensureCIRIntType(builder, loc, cirValue, cirIntType);
  }

  // === Estrazione dell'upper bound e dell'operatore di confronto ===

  const auto *condBinOp = dyn_cast_or_null<BinaryOperator>(forStmt->getCond());
  if (!condBinOp)
    return mlir::failure();

  BinaryOperatorKind opKind = condBinOp->getOpcode();

  // Determina quale lato del confronto contiene il bound.
  // La variabile del loop può apparire su entrambi i lati:
  //   `i < ub`, `i <= ub`, `i > lb`, `i >= lb` (variabile a sinistra)
  //   `ub > i`, `ub >= i`, `lb < i`, `lb <= i` (variabile a destra)
  const Expr *boundExpr = nullptr;
  if (opKind != BO_LT && opKind != BO_LE && opKind != BO_GT &&
      opKind != BO_GE)
    return mlir::failure(); // Operatore non supportato (es. !=, ==)

  const auto *lhsRef = dyn_cast<DeclRefExpr>(
      condBinOp->getLHS()->IgnoreParenImpCasts());
  bool varOnLHS = lhsRef && lhsRef->getDecl() == varDecl;
  boundExpr = varOnLHS ? condBinOp->getRHS() : condBinOp->getLHS();
  inclusive = (opKind == BO_LE || opKind == BO_GE);

  // Genera il valore per l'upper bound (costante o espressione).
  if (auto constVal = getIntLiteralValue(boundExpr)) {
    upperBound = builder.getConstInt(loc, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = cgf.emitScalarExpr(boundExpr);
    upperBound = ensureCIRIntType(builder, loc, cirValue, cirIntType);
  }

  // === Estrazione dello step (passo di incremento) ===

  // Caso 1: operatore unario (i++ → step=1, i-- → step=-1)
  if (const auto *unaryOp =
          dyn_cast_or_null<UnaryOperator>(forStmt->getInc())) {
    int64_t val = unaryOp->isIncrementOp() ? 1 : -1;
    step = builder.getConstInt(loc, cirIntType, val);
  }
  // Caso 2: operatore binario (i += step, i = i + step, ecc.)
  else if (const auto *binOp =
                 dyn_cast_or_null<BinaryOperator>(forStmt->getInc())) {
    const Expr *stepExpr = nullptr;

    if (binOp->isCompoundAssignmentOp()) {
      // `i += step` → lo step è il lato destro
      stepExpr = binOp->getRHS();
    } else if (binOp->isAssignmentOp()) {
      // `i = i + step` o `i = step + i`
      // Bisogna capire quale operando è la variabile e quale lo step.
      if (auto *subBinOp =
              dyn_cast<BinaryOperator>(binOp->getRHS()->IgnoreImpCasts())) {
        const Expr *lhs = subBinOp->getLHS()->IgnoreImpCasts();
        const Expr *rhs = subBinOp->getRHS()->IgnoreImpCasts();
        // Identifica quale operando è la variabile di induzione
        // e restituisce l'altro come step.
        if (auto *lhsRef = dyn_cast<DeclRefExpr>(lhs)) {
          stepExpr = (lhsRef->getDecl() == varDecl) ? rhs : lhs;
        } else if (auto *rhsRef = dyn_cast<DeclRefExpr>(rhs)) {
          stepExpr = (rhsRef->getDecl() == varDecl) ? lhs : rhs;
        }
      }
    }

    // Genera il valore per lo step (costante o espressione).
    if (stepExpr) {
      if (auto constVal = getIntLiteralValue(stepExpr)) {
        step = builder.getConstInt(loc, cirIntType, *constVal);
      } else {
        mlir::Value cirValue = cgf.emitScalarExpr(stepExpr);
        step = ensureCIRIntType(builder, loc, cirValue, cirIntType);
      }
    }
  }

  // Step di default: 1 (se non è stato riconosciuto il pattern di incremento).
  if (!step)
    step = builder.getConstInt(loc, cirIntType, 1);

  // === Emissione dell'init del loop e allocazione della IV ===

  // Caso DeclStmt (`int i = 0`): l'emissione crea naturalmente l'alloca.
  if (const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit())) {
    if (cgf.emitStmt(declStmt, /*useCurrentScope=*/true).failed())
      return mlir::failure();
  }
  // Caso assignment (`i = 0`): la variabile è dichiarata fuori dal loop.
  // OpenMP richiede che la variabile di induzione sia implicitamente
  // privata, quindi creiamo una nuova alloca privata dentro la regione
  // corrente e rimappiamo localDeclMap per usarla.
  else if (forStmt->getInit()) {
    // Salva l'indirizzo originale per ripristinarlo dopo il loop
    savedAddr = cgf.getAddrOfLocalVar(varDecl);
    // Crea un'alloca temporanea per la copia privata della IV
    Address privateAddr =
        cgf.createMemTemp(loopVarQType, loc, varDecl->getName() + ".iv");
    // Inizializza la copia privata con il lower bound
    cir::StoreOp::create(builder, loc, lowerBound, privateAddr.getPointer(),
                         /*is_volatile=*/nullptr, /*alignment=*/nullptr,
                         /*sync_scope=*/nullptr, /*mem_order=*/nullptr);
    // Rimappa la variabile alla copia privata
    cgf.replaceAddrOfLocalVar(varDecl, privateAddr);
  }

  // === Conversione dei limiti da CIR int a standard MLIR int ===
  // L'op omp.loop_nest richiede operandi di tipo IntLikeType,
  // non tipi CIR. Inserisce UnrealizedConversionCastOp.
  mlir::Value stdLB = cirIntToStdInt(builder, loc, lowerBound);
  mlir::Value stdUB = cirIntToStdInt(builder, loc, upperBound);
  mlir::Value stdStep = cirIntToStdInt(builder, loc, step);
  mlir::Type loopBoundsType = stdLB.getType(); // Il tipo delle IV nel loop_nest

  // Aggiunge i limiti e i metadati alla struttura LoopBounds.
  bounds.lowerBounds.push_back(stdLB);
  bounds.upperBounds.push_back(stdUB);
  bounds.steps.push_back(stdStep);
  bounds.inductionVarTypes.push_back(loopBoundsType);
  bounds.inductionVars.push_back(varDecl);
  bounds.inclusive.push_back(inclusive);
  bounds.savedInductionVarAddrs.push_back(savedAddr);
  return mlir::success();
}

/// Estrae i limiti di uno o più ForStmt annidati (per il supporto collapse).
///
/// Per il caso non-collapsed (numLoops=1), processa un singolo ForStmt.
/// Per il caso collapsed (numLoops>1), usa
/// OMPLoopBasedDirective::doForAllLoops per traversare N loop annidati
/// perfettamente, estraendo i limiti di ciascuno.
///
/// In caso di collapse, memorizza anche il body del loop più interno
/// (innermostBody) affinché emitForStmt emetta solo quello e non
/// l'intero albero dei for annidati (che ora sono parte del loop_nest).
///
/// Popola currentOMPLoopBounds, che sarà consumato da emitForStmt.
mlir::LogicalResult CIRGenFunction::extractOMPLoopBounds(
    const ForStmt *forStmt, mlir::Location loc, unsigned numLoops) {

  LoopBounds bounds;
  bounds.numLoops = numLoops;

  if (numLoops == 1) {
    // Caso semplice (non-collapsed): un solo loop.
    if (extractSingleLoopBounds(*this, builder, forStmt, loc, bounds).failed())
      return mlir::failure();
  } else {
    // Caso collapsed: traversa N loop annidati perfettamente.
    bool failed = false;
    const ForStmt *innermostFor = nullptr;

    // doForAllLoops è un helper di Clang che visita i loop annidati
    // in una direttiva OpenMP con clausola collapse(N).
    // La callback viene chiamata per ogni livello di annidamento.
    OMPLoopBasedDirective::doForAllLoops(
        const_cast<Stmt *>(cast<Stmt>(forStmt)),
        /*TryImperfectlyNestedLoops=*/false, numLoops,
        [&](unsigned /*idx*/, Stmt *curStmt) -> bool {
          auto *innerFor = dyn_cast<ForStmt>(curStmt);
          if (!innerFor) {
            failed = true;
            return true; // stop — non è un ForStmt
          }
          // Estrae i limiti di questo livello e li aggiunge a bounds
          if (extractSingleLoopBounds(*this, builder, innerFor, loc, bounds)
                  .failed()) {
            failed = true;
            return true; // stop — estrazione fallita
          }
          innermostFor = innerFor; // Aggiorna il loop più interno
          return false; // continue al prossimo livello
        });
    if (failed)
      return mlir::failure();

    // Per i loop collapsed, il body da emettere è quello del loop più
    // interno. I loop esterni sono ora "assorbiti" nell'omp.loop_nest
    // con operandi multi-dimensionali (lbs[], ubs[], steps[]).
    bounds.innermostBody = innermostFor->getBody();
  }

  // Salva i limiti in currentOMPLoopBounds, un campo optional<LoopBounds>
  // di CIRGenFunction. Sarà consumato da emitForStmt per creare
  // l'omp.loop_nest.
  currentOMPLoopBounds = std::move(bounds);
  return mlir::success();
}

// =====================================================================
// emitOMPForDirective — #pragma omp for
// =====================================================================

/// Emette una direttiva `#pragma omp for { for(i=...) { body } }`.
///
/// Questa direttiva genera una struttura a 2 livelli:
///   omp.wsloop {          ← creato qui
///     omp.loop_nest {     ← creato da emitForStmt (in CIRGenStmt.cpp)
///       // body del loop
///       omp.yield
///     }
///   }
///
/// Il vincolo chiave è che omp.wsloop deve contenere esattamente una op
/// innestata (il loop_nest). Per questo motivo, il remapping delle
/// variabili private/ridotte viene fatto nel body del loop_nest anziché
/// nel body del wsloop. I processori (dsp, rdp) sono salvati come
/// "deferred" nei campi currentOMPDataSharingProcessor e
/// currentOMPReductionProcessor di CIRGenFunction, e applicati
/// quando emitForStmt crea il loop_nest.
mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &s) {

  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());

  // Recupera il ForStmt C/C++ dal CapturedStmt della direttiva OpenMP.
  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (!forStmt)
    return mlir::failure();

  // Numero di loop da collassare (1 se nessuna clausola collapse).
  unsigned numLoops = s.getLoopsNumber();

  // Estrae i limiti del loop (o dei loop per collapse), emette l'init
  // (alloca per la IV) e popola currentOMPLoopBounds.
  if (extractOMPLoopBounds(forStmt, begin, numLoops).failed())
    return mlir::failure();

  // Crea l'op omp.wsloop (worksharing loop).
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto wsloopOp = mlir::omp::WsloopOp::create(builder, begin, retTy, operands);

  // Processa le clausole non gestite altrove (schedule, ecc.).
  emitOpenMPClauses(wsloopOp, s.clauses());

  // === Data Sharing ===
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, wsloopOp);

  if (dsp.hasPrivateVars()) {
    wsloopOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    wsloopOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // === Riduzione ===
  OMPReductionClauseOps redClauseOps;
  OMPReductionProcessor rdp(*this, builder, begin);
  rdp.processReductionVars(s.clauses(), redClauseOps, wsloopOp);

  if (rdp.hasReductionVars()) {
    wsloopOp.getReductionVarsMutable().append(redClauseOps.reductionVars);
    wsloopOp.setReductionSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), redClauseOps.reductionSyms));
    wsloopOp.setReductionByrefAttr(
        mlir::DenseBoolArrayAttr::get(builder.getContext(),
                                      redClauseOps.reductionByref));
  }

  // === Creazione del blocco della regione wsloop ===
  // Aggiunge block arguments per private e reduction vars.
  // NOTA: il remapping (cast e remap) non viene fatto qui ma nel
  // body del loop_nest (via currentOMPDataSharingProcessor), perché
  // il wsloop deve contenere esattamente una op innestata.
  mlir::Region &region = wsloopOp.getRegion();
  mlir::Block *block = new mlir::Block();
  region.push_back(block);
  dsp.addBlockArgs(*block);
  rdp.addBlockArgs(*block);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  // Salva i puntatori ai processori per il remapping differito.
  // emitForStmt li userà per applicare il remapping nel body del loop_nest.
  assert(!currentOMPDataSharingProcessor &&
         "nested wsloop privatization not supported");
  currentOMPDataSharingProcessor = &dsp;
  currentOMPReductionProcessor = &rdp;

  // Salva le info di ripristino PRIMA di emitStmt, che cancella
  // currentOMPLoopBounds dentro emitForStmt per evitare che i
  // for-loop interni al body vengano trattati come ulteriori loop_nest.
  llvm::SmallVector<Address, 2> savedAddrs =
      currentOMPLoopBounds->savedInductionVarAddrs;
  llvm::SmallVector<const VarDecl *, 2> inductionVars =
      currentOMPLoopBounds->inductionVars;

  // Emette il ForStmt: questo crea l'omp.loop_nest come unica op
  // innestata nel wsloop. Il remapping delle variabili private/ridotte
  // avviene dentro il body del loop_nest.
  // NOTA: currentOMPLoopBounds viene cancellato dentro emitForStmt
  // dopo la creazione del loop_nest.
  if (emitStmt(forStmt, /*useCurrentScope=*/false).failed())
    res = mlir::failure();

  // Reset dei puntatori ai processori dopo l'emissione.
  currentOMPDataSharingProcessor = nullptr;
  currentOMPReductionProcessor = nullptr;

  // Ripristina gli indirizzi originali per le variabili di induzione
  // che erano state implicitamente privatizzate (dichiarate fuori dal for).
  for (unsigned i = 0; i < inductionVars.size(); ++i) {
    if (savedAddrs[i].isValid())
      replaceAddrOfLocalVar(inductionVars[i], savedAddrs[i]);
  }

  return res;
}

// =====================================================================
// Stub NYI per direttive semplici
// =====================================================================

// =====================================================================
// emitOMPTaskwaitDirective — #pragma omp taskwait
// =====================================================================

/// Emette una direttiva `#pragma omp taskwait`.
///
/// Genera un'op `omp.taskwait` che attende il completamento dei task
/// figli generati dal task corrente. Le clausole depend/nowait non
/// sono ancora supportate.
mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  if (!s.clauses().empty()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "OpenMP taskwait with clauses");
    return mlir::failure();
  }
  // Il builder basato su TaskwaitOperands inizializza correttamente
  // l'attributo operandSegmentSizes (richiesto da AttrSizedOperandSegments).
  mlir::omp::TaskwaitOp::create(builder, getLoc(s.getBeginLoc()),
                                mlir::omp::TaskwaitOperands());
  return mlir::success();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskyieldDirective");
  return mlir::failure();
}

// =====================================================================
// emitOMPBarrierDirective — #pragma omp barrier
// =====================================================================

/// Emette una direttiva `#pragma omp barrier`.
///
/// Genera un'op `omp.barrier` che sincronizza tutti i thread del team.
/// Nessuna clausola è supportata per barrier (asserzione di sicurezza).
mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::omp::BarrierOp::create(builder, getLoc(s.getBeginLoc()));
  assert(s.clauses().empty() && "omp barrier doesn't support clauses");
  return mlir::success();
}

// =====================================================================
// Stub NYI per direttive varie
// =====================================================================

mlir::LogicalResult
CIRGenFunction::emitOMPMetaDirective(const OMPMetaDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMetaDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCanonicalLoop(const OMPCanonicalLoop &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCanonicalLoop");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSimdDirective(const OMPSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTileDirective(const OMPTileDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTileDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPUnrollDirective(const OMPUnrollDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPUnrollDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFuseDirective(const OMPFuseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFuseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPForSimdDirective(const OMPForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionsDirective(const OMPSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionDirective(const OMPSectionDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionDirective");
  return mlir::failure();
}

// =====================================================================
// emitOMPSingleDirective — #pragma omp single
// =====================================================================

/// Emette una direttiva `#pragma omp single { body }`.
///
/// Genera un'op `omp.single` che esegue il body su un solo thread.
/// Supporta le clausole:
///   - nowait: salta la barriera implicita alla fine della regione
///   - private/firstprivate: tramite OMPDataSharingProcessor
///
/// Struttura IR risultante:
///   omp.single nowait private(@x.privatizer %x) {
///     // ... body ...
///     omp.terminator
///   }
mlir::LogicalResult
CIRGenFunction::emitOMPSingleDirective(const OMPSingleDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  mlir::omp::SingleOperands clauseOps;

  // Gestione della clausola nowait.
  // nowait elimina la barriera implicita alla fine dell'omp single.
  // È rappresentata come un UnitAttr (presente = nowait, assente = wait).
  for (const OMPClause *c : s.clauses()) {
    if (isa<OMPNowaitClause>(c))
      clauseOps.nowait = builder.getUnitAttr();
  }

  // Crea l'op omp.single con gli operandi della clausola nowait.
  auto singleOp =
      mlir::omp::SingleOp::create(builder, begin, clauseOps);

  // === Data Sharing ===
  OMPPrivateClauseOps privClauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), privClauseOps, singleOp);

  if (dsp.hasPrivateVars()) {
    singleOp.getPrivateVarsMutable().append(privClauseOps.privateVars);
    singleOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), privClauseOps.privateSyms));
  }

  // === Regione e body ===
  {
    mlir::Block &block = singleOp.getRegion().emplaceBlock();
    dsp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);

    // Remapping RAII delle variabili private
    auto remapGuard = dsp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    // Recupera il body, srotolando il CapturedStmt se presente.
    const Stmt *bodyStmt = s.getAssociatedStmt();
    if (const auto *cs = dyn_cast<CapturedStmt>(bodyStmt))
      bodyStmt = cs->getCapturedStmt();

    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}

// =====================================================================
// emitOMPMasterDirective — #pragma omp master
// =====================================================================

/// Emette una direttiva `#pragma omp master { body }`.
///
/// Genera un'op `omp.master` che esegue il body solo sul thread master
/// (thread ID 0). Non supporta clausole.
///
/// Struttura IR risultante:
///   omp.master {
///     // ... body ...
///     omp.terminator
///   }
mlir::LogicalResult
CIRGenFunction::emitOMPMasterDirective(const OMPMasterDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // Crea l'op omp.master (nessun operando — nessuna clausola supportata).
  auto masterOp = mlir::omp::MasterOp::create(builder, begin);

  {
    // Crea un blocco vuoto nella regione del master.
    mlir::Block &block = masterOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    // Recupera e srotola il body dal CapturedStmt.
    const Stmt *bodyStmt = s.getAssociatedStmt();
    if (const auto *cs = dyn_cast<CapturedStmt>(bodyStmt))
      bodyStmt = cs->getCapturedStmt();

    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}

// =====================================================================
// Stub NYI
// =====================================================================

mlir::LogicalResult
CIRGenFunction::emitOMPCriticalDirective(const OMPCriticalDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCriticalDirective");
  return mlir::failure();
}

// =====================================================================
// emitOMPParallelForDirective — #pragma omp parallel for
// =====================================================================

/// Emette una direttiva `#pragma omp parallel for { for(i=...) { body } }`.
///
/// Questa è una direttiva composta che fonde parallel e for in una
/// singola operazione del runtime. Genera una struttura a 3 livelli:
///
///   omp.parallel private(...) reduction(...) {
///     // block args per private/reduction vars
///     // remapping RAII delle variabili
///     omp.wsloop schedule(...) {
///       omp.loop_nest lb ub step {
///         // IV store + body
///         omp.yield
///       }
///     }
///     omp.terminator
///   }
///
/// Le variabili private e di riduzione vanno sull'op parallel (non
/// sul wsloop), perché la privatizzazione avviene a livello di team,
/// non a livello di iterazione.
///
/// Le clausole schedule vanno sul wsloop (non sul parallel), perché
/// controllano la distribuzione delle iterazioni tra i thread.
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(const OMPParallelForDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // Recupera il ForStmt dal CapturedStmt.
  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (!forStmt)
    return mlir::failure();

  // Funzionalità non ancora supportate.
  if (s.hasCancel())
    getCIRGenModule().errorNYI(s.getBeginLoc(),
                               "OpenMP ParallelFor with Cancel");
  if (s.getTaskReductionRefExpr())
    getCIRGenModule().errorNYI(s.getBeginLoc(),
                               "OpenMP ParallelFor with Task Reduction");

  // === Op parallel esterna ===
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto parallelOp =
      mlir::omp::ParallelOp::create(builder, begin, retTy, operands);

  // Processa clausole a livello parallel (proc_bind, num_threads, ecc.).
  emitOpenMPClauses(parallelOp, s.clauses());

  // Data sharing: le variabili private vanno sull'op parallel.
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, parallelOp);

  if (dsp.hasPrivateVars()) {
    parallelOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    parallelOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // Riduzione: anche le variabili di riduzione vanno sull'op parallel.
  OMPReductionClauseOps redClauseOps;
  OMPReductionProcessor rdp(*this, builder, begin);
  rdp.processReductionVars(s.clauses(), redClauseOps, parallelOp);

  if (rdp.hasReductionVars()) {
    parallelOp.getReductionVarsMutable().append(redClauseOps.reductionVars);
    parallelOp.setReductionSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), redClauseOps.reductionSyms));
    parallelOp.setReductionByrefAttr(
        mlir::DenseBoolArrayAttr::get(builder.getContext(),
                                      redClauseOps.reductionByref));
  }

  // === Regione parallel con remapping e wsloop innestato ===
  {
    mlir::Block &block = parallelOp.getRegion().emplaceBlock();
    dsp.addBlockArgs(block);
    rdp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    // Remapping RAII: cast block args → CIR pointers e remap localDeclMap.
    // Questo avviene a livello parallel (non wsloop), perché le variabili
    // sono private a livello di thread, non di iterazione.
    auto remapGuard = dsp.applyRemapping();
    auto redRemapGuard = rdp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    // Numero di loop da collassare.
    unsigned numLoops = s.getLoopsNumber();

    // Estrae i limiti del loop DENTRO la regione parallel (importante:
    // le alloca per le IV devono essere dentro la regione parallel
    // per essere thread-private).
    if (extractOMPLoopBounds(forStmt, begin, numLoops).failed())
      return mlir::failure();

    // === Op wsloop interna ===
    llvm::SmallVector<mlir::Type> wsRetTy;
    llvm::SmallVector<mlir::Value> wsOperands;
    auto wsloopOp =
        mlir::omp::WsloopOp::create(builder, begin, wsRetTy, wsOperands);

    // Processa clausole a livello wsloop (schedule, ecc.).
    // Le clausole proc_bind e private sono no-op qui (già gestite sopra).
    emitOpenMPClauses(wsloopOp, s.clauses());

    // Crea il blocco del wsloop. NON ha block arguments per private/reduction
    // perché quelli sono gestiti a livello parallel.
    mlir::Region &wsRegion = wsloopOp.getRegion();
    mlir::Block *wsBlock = new mlir::Block();
    wsRegion.push_back(wsBlock);

    // Salva le info di ripristino per le IV implicitamente privatizzate.
    llvm::SmallVector<Address, 2> savedAddrs =
        currentOMPLoopBounds->savedInductionVarAddrs;
    llvm::SmallVector<const VarDecl *, 2> inductionVars =
        currentOMPLoopBounds->inductionVars;

    {
      mlir::OpBuilder::InsertionGuard wsGuard(builder);
      builder.setInsertionPointToStart(wsBlock);

      // Emette il ForStmt che crea l'omp.loop_nest come unica op
      // innestata nel wsloop. Non serve remapping differito qui:
      // le variabili private/ridotte sono già rimappate a livello parallel.
      // NOTA: currentOMPLoopBounds viene cancellato dentro emitForStmt.
      if (emitStmt(forStmt, /*useCurrentScope=*/false).failed())
        res = mlir::failure();
    }

    // Ripristina gli indirizzi originali delle IV implicitamente privatizzate.
    for (unsigned i = 0; i < inductionVars.size(); ++i) {
      if (savedAddrs[i].isValid())
        replaceAddrOfLocalVar(inductionVars[i], savedAddrs[i]);
    }

    // Termina la regione parallel con omp.terminator.
    mlir::omp::TerminatorOp::create(builder, end);
  }
  // Qui remapGuard e redRemapGuard vengono distrutte → ripristino mappature.

  return res;
}

// =====================================================================
// Stub NYI per tutte le altre direttive OpenMP
// =====================================================================

mlir::LogicalResult CIRGenFunction::emitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterDirective(
    const OMPParallelMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelSectionsDirective");
  return mlir::failure();
}
// =====================================================================
// emitOMPTaskDirective — #pragma omp task
// =====================================================================

/// Emette una direttiva `#pragma omp task { body }`.
///
/// Genera un'op `omp.task` con:
///   - Clausole semplici (if, final, priority, untied, mergeable)
///     tramite emitOpenMPClauses
///   - Variabili private/firstprivate tramite OMPDataSharingProcessor
///
/// NOTA sul data sharing implicito: per i task, le variabili referenziate
/// nel body che non sono shared nel contesto esterno sono firstprivate
/// per default (OpenMP spec). Non serve gestirlo esplicitamente qui:
/// Sema aggiunge clausole OMPFirstprivateClause IMPLICITE a s.clauses()
/// per le variabili catturate, quindi il DataSharingProcessor le
/// raccoglie come se fossero state scritte dall'utente.
///
/// Struttura IR risultante (analoga a quella generata da Flang):
///   omp.task private(@x.privatizer %x -> %arg0 : !llvm.ptr) {
///     // cast !llvm.ptr → !cir.ptr<T>
///     // ... body emesso con variabili rimappate ...
///     omp.terminator
///   }
mlir::LogicalResult
CIRGenFunction::emitOMPTaskDirective(const OMPTaskDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // Crea l'op omp.task tramite il builder basato su TaskOperands:
  // a differenza del builder generico, inizializza correttamente
  // l'attributo operandSegmentSizes (TaskOp ha AttrSizedOperandSegments),
  // necessario per poter aggiungere operandi variadici in seguito
  // con le mutable operand ranges.
  auto taskOp =
      mlir::omp::TaskOp::create(builder, begin, mlir::omp::TaskOperands());

  // Processa le clausole semplici (if, final, priority, untied,
  // mergeable). Le clausole private/firstprivate sono no-op qui:
  // vengono gestite dal DataSharingProcessor subito sotto.
  emitOpenMPClauses(taskOp, s.clauses());

  // === Data Sharing: private/firstprivate (esplicite e implicite) ===
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, taskOp);

  if (dsp.hasPrivateVars()) {
    taskOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    taskOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // === Creazione della regione e emissione del body ===
  {
    mlir::Block &block = taskOp.getRegion().emplaceBlock();

    // Block arguments !llvm.ptr per le variabili private: riceveranno
    // i puntatori alle copie task-local dal runtime OpenMP.
    dsp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);

    // Remapping RAII: nel body, le variabili puntano alle copie private.
    auto remapGuard = dsp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    if (s.hasCancel())
      getCIRGenModule().errorNYI(s.getBeginLoc(), "OpenMP Task with Cancel");

    // Recupera e srotola il body dal CapturedStmt.
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_task);
    const Stmt *bodyStmt = cs->getCapturedStmt();

    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskgroupDirective(const OMPTaskgroupDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskgroupDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFlushDirective(const OMPFlushDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFlushDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDepobjDirective(const OMPDepobjDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDepobjDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPScanDirective(const OMPScanDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScanDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPOrderedDirective(const OMPOrderedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPOrderedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAtomicDirective(const OMPAtomicDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAtomicDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDirective(const OMPTargetDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTargetDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDirective(const OMPTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPCancellationPointDirective(
    const OMPCancellationPointDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPCancellationPointDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCancelDirective(const OMPCancelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCancelDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDataDirective(const OMPTargetDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetEnterDataDirective(
    const OMPTargetEnterDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetEnterDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetExitDataDirective(
    const OMPTargetExitDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetExitDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelDirective(
    const OMPTargetParallelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForDirective(
    const OMPTargetParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskLoopDirective(const OMPTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTaskLoopSimdDirective(
    const OMPTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopDirective(
    const OMPMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopSimdDirective(
    const OMPMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopDirective(
    const OMPMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopSimdDirective(
    const OMPMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelGenericLoopDirective(
    const OMPParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedDirective(
    const OMPParallelMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopDirective(
    const OMPParallelMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopSimdDirective(
    const OMPParallelMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopDirective(
    const OMPParallelMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopSimdDirective(
    const OMPParallelMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDistributeDirective(const OMPDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForDirective(
    const OMPDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForSimdDirective(
    const OMPDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeSimdDirective(
    const OMPDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelGenericLoopDirective(
    const OMPTargetParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForSimdDirective(
    const OMPTargetParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetSimdDirective(const OMPTargetSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsGenericLoopDirective(
    const OMPTargetTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetUpdateDirective(
    const OMPTargetUpdateDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetUpdateDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeDirective(
    const OMPTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeSimdDirective(
    const OMPTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDistributeParallelForSimdDirective(
    const OMPTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(), "OpenMP OMPTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeParallelForDirective(
    const OMPTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsGenericLoopDirective(
    const OMPTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDirective(const OMPTargetTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeDirective(
    const OMPTargetTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForDirective(
    const OMPTargetTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForSimdDirective(
    const OMPTargetTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeSimdDirective(
    const OMPTargetTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInteropDirective(const OMPInteropDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPInteropDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDispatchDirective(const OMPDispatchDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDispatchDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPGenericLoopDirective(const OMPGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPReverseDirective(const OMPReverseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPReverseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInterchangeDirective(const OMPInterchangeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPInterchangeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAssumeDirective(const OMPAssumeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAssumeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMaskedDirective(const OMPMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPStripeDirective(const OMPStripeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPStripeDirective");
  return mlir::failure();
}
