//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supporto al data sharing (privatizzazione) e alla riduzione OpenMP
// per la codegen CIR.
//
// Questo header dichiara le classi OMPDataSharingProcessor e
// OMPReductionProcessor, che gestiscono rispettivamente le clausole
// private/firstprivate e reduction durante la generazione di codice CIR
// per direttive OpenMP.
//
// Il problema principale che risolvono è il "bridge" fra due dialetti MLIR:
//   - CIR usa puntatori tipizzati: !cir.ptr<T>
//   - Il dialetto OMP usa puntatori opachi LLVM: !llvm.ptr
// Le classi gestiscono i cast tra i due sistemi di tipi e il remapping
// delle variabili locali affinché il body delle regioni OpenMP usi
// le copie private/ridotte anziché le variabili originali.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H

#include "CIRGenBuilder.h"
#include "CIRGenValue.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

// Forward declarations dei tipi Clang usati nelle interfacce.
namespace clang {
class OMPClause;  // Classe base per tutte le clausole OpenMP (private, reduction, ecc.)
class VarDecl;    // Dichiarazione di variabile C/C++
} // namespace clang

namespace clang::CIRGen {

class CIRGenFunction; // Classe che gestisce la codegen a livello di funzione

// ===================================================================
// Strutture dati per la privatizzazione (private / firstprivate)
// ===================================================================

/// Operandi accumulati da passare all'op OpenMP per la privatizzazione.
/// Dopo processStep1(), questi vettori contengono i valori e i simboli
/// necessari per popolare gli attributi "private_vars" e "privatizers"
/// dell'op (es. omp.parallel, omp.wsloop).
struct OMPPrivateClauseOps {
  llvm::SmallVector<mlir::Value> privateVars;    // Puntatori !llvm.ptr alle variabili originali
  llvm::SmallVector<mlir::Attribute> privateSyms; // Riferimenti simbolici (@nome.privatizer) alle omp.private ops
};

/// Metadati per singola variabile privata, raccolti durante l'elaborazione
/// delle clausole. Ogni entry traccia la variabile AST, il suo indirizzo
/// originale nel dialetto CIR, il tipo dell'elemento, il nome del
/// privatizer e il block argument (riempito più tardi da addBlockArgs).
struct OMPPrivateVarEntry {
  const VarDecl *varDecl;       // La dichiarazione di variabile nell'AST Clang
  mlir::Value originalAddr;     // Indirizzo originale come !cir.ptr<T> (l'alloca)
  mlir::Type elementType;       // Tipo dell'elemento CIR (es. !cir.int<s, 32>)
  std::string privatizerName;   // Nome simbolico dell'omp.private op (es. "x.privatizer")
  mlir::Value blockArg;         // Block argument !llvm.ptr, riempito da addBlockArgs()
};

/// Processore centralizzato per il data sharing OpenMP
/// (private/firstprivate/lastprivate) nella codegen CIR.
///
/// Il design è ispirato al DataSharingProcessor di Flang e segue
/// un protocollo in 3 fasi:
///
///   // Fase 1: raccoglie variabili, crea op a livello di modulo,
///   //         popola gli operandi per l'op OpenMP
///   OMPDataSharingProcessor dsp(cgf, builder, loc);
///   dsp.processStep1(clauses, clauseOps, op);
///
///   // ... ora si attaccano clauseOps all'op OpenMP ...
///
///   // Fase 2: aggiunge block arguments alla regione dell'op
///   dsp.addBlockArgs(block);
///
///   // Fase 3: rimappa le variabili locali alle copie private
///   {
///     auto guard = dsp.applyRemapping();
///     // ... qui si emette il body — le variabili puntano alle copie private ...
///   }
///   // Quando guard esce dallo scope, le mappature originali sono ripristinate.
class OMPDataSharingProcessor {
public:
  /// Guardia RAII che ripristina le entry del localDeclMap alla distruzione.
  ///
  /// Quando applyRemapping() rimappa una variabile alla sua copia privata,
  /// salva la mappatura precedente in questa guardia. Quando la guardia
  /// viene distrutta (uscita dallo scope), le mappature originali vengono
  /// ripristinate automaticamente, così il codice successivo al body
  /// OpenMP vede di nuovo le variabili originali.
  class RemapGuard {
    CIRGenFunction &cgf; // Riferimento al generatore di codice per la funzione
    llvm::SmallVector<std::pair<const VarDecl *, Address>> savedAddrs; // Mappature salvate (var → indirizzo originale)

  public:
    // Costruttore: prende il CGF e le mappature salvate da ripristinare
    RemapGuard(CIRGenFunction &cgf,
               llvm::SmallVector<std::pair<const VarDecl *, Address>> saved);
    // Distruttore: ripristina tutte le mappature salvate nel localDeclMap
    ~RemapGuard();
    // Move constructor: permette di restituire la guardia per valore
    RemapGuard(RemapGuard &&other) noexcept;
    // Operazioni di copia/assegnamento disabilitate — la guardia è unica
    RemapGuard &operator=(RemapGuard &&) = delete;
    RemapGuard(const RemapGuard &) = delete;
    RemapGuard &operator=(const RemapGuard &) = delete;
  };

  // Costruttore: inizializza con il generatore di funzione, il builder MLIR
  // e la source location da usare per le op generate.
  OMPDataSharingProcessor(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                          mlir::Location loc);

  /// Fase 1: raccoglie le variabili private dalle clausole, crea le op
  /// omp.private a livello di modulo, e popola clauseOps con i cast
  /// !cir.ptr → !llvm.ptr inseriti prima di \p insertBeforeOp
  /// (necessario per rispettare la dominanza SSA in MLIR).
  void processStep1(llvm::ArrayRef<const OMPClause *> clauses,
                    OMPPrivateClauseOps &clauseOps,
                    mlir::Operation *insertBeforeOp);

  /// Fase 2: aggiunge un block argument di tipo !llvm.ptr al \p block
  /// per ogni variabile privata. Questi block arguments sono i "parametri"
  /// della regione OpenMP che riceveranno le copie private dal runtime.
  void addBlockArgs(mlir::Block &block);

  /// Fase 3: inserisce cast !llvm.ptr → !cir.ptr al punto di inserimento
  /// corrente e rimappa le entry di localDeclMap affinché il body usi
  /// le copie private. Restituisce una guardia RAII che ripristina
  /// le mappature originali alla distruzione.
  RemapGuard applyRemapping();

  /// Restituisce true se ci sono variabili private da gestire.
  bool hasPrivateVars() const { return !entries.empty(); }

  /// Accesso in sola lettura alle entry raccolte.
  llvm::ArrayRef<OMPPrivateVarEntry> getEntries() const { return entries; }

private:
  CIRGenFunction &cgf;                        // Generatore di codice per la funzione corrente
  CIRGenBuilderTy &builder;                   // Builder MLIR per creare operazioni
  mlir::Location loc;                         // Source location per diagnostica/debugging
  llvm::SmallVector<OMPPrivateVarEntry> entries; // Entry raccolte (una per variabile privata)

  /// Converte un tipo CIR (es. !cir.int<s,32>, !cir.float) nel
  /// corrispondente tipo standard MLIR (es. i32, f32) per l'uso
  /// nelle regioni delle op omp.private.
  mlir::Type convertCIRTypeToStdType(mlir::Type cirType);

  /// Crea (o riusa se già esistente) un'op omp.private a livello di modulo.
  /// L'op contiene le regioni init e copy che descrivono come inizializzare
  /// e copiare la variabile privata.
  void getOrCreatePrivateOp(llvm::StringRef name, mlir::Type stdType,
                            mlir::omp::DataSharingClauseType dsType);
};

// ===================================================================
// Strutture dati per la riduzione (reduction)
// ===================================================================

/// Tipi di operatore di riduzione supportati per OpenMP.
/// Ciascun valore corrisponde a un operatore C/C++ usato nella
/// clausola reduction(op:var).
enum class OMPReductionKind {
  Add,        // + (e anche -, che usa lo stesso combiner)
  Multiply,   // *
  BitwiseAnd, // &
  BitwiseOr,  // |
  BitwiseXor, // ^
  LogicalAnd, // &&
  LogicalOr,  // ||
};

/// Operandi accumulati da passare all'op OpenMP per la riduzione.
/// Analogo a OMPPrivateClauseOps ma per le clausole reduction.
struct OMPReductionClauseOps {
  llvm::SmallVector<mlir::Value> reductionVars;   // Puntatori !llvm.ptr alle variabili di riduzione
  llvm::SmallVector<mlir::Attribute> reductionSyms; // Riferimenti simbolici alle omp.declare_reduction ops
  llvm::SmallVector<bool> reductionByref;          // Se la riduzione è by-reference (false per scalari)
};

/// Metadati per singola variabile di riduzione. Struttura analoga
/// a OMPPrivateVarEntry ma per il contesto reduction.
struct OMPReductionVarEntry {
  const VarDecl *varDecl;       // Dichiarazione di variabile nell'AST
  mlir::Value originalAddr;     // Indirizzo originale !cir.ptr<T>
  mlir::Type elementType;       // Tipo elemento CIR (T)
  std::string reductionName;    // Nome simbolico dell'omp.declare_reduction (es. "add_x")
  mlir::Value blockArg;         // Block argument !llvm.ptr, riempito da addBlockArgs()
};

/// Processore per le clausole reduction OpenMP nella codegen CIR.
///
/// Segue lo stesso protocollo in 3 fasi del DataSharingProcessor:
/// 1. processReductionVars() — raccoglie variabili, crea omp.declare_reduction
/// 2. addBlockArgs() — aggiunge block arguments alla regione
/// 3. applyRemapping() — rimappa localDeclMap alle copie thread-local
///
/// La differenza principale rispetto alla privatizzazione è che le op
/// omp.declare_reduction contengono anche una regione "combiner" che
/// descrive come combinare i risultati parziali dei thread (es. somma).
class OMPReductionProcessor {
public:
  OMPReductionProcessor(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                        mlir::Location loc);

  /// Fase 1: raccoglie le variabili di riduzione dalle clausole,
  /// crea le op omp.declare_reduction a livello di modulo, e popola
  /// clauseOps con i cast e i riferimenti simbolici.
  void processReductionVars(llvm::ArrayRef<const OMPClause *> clauses,
                            OMPReductionClauseOps &clauseOps,
                            mlir::Operation *insertBeforeOp);

  /// Fase 2: aggiunge block arguments !llvm.ptr per ogni variabile
  /// di riduzione.
  void addBlockArgs(mlir::Block &block);

  /// Fase 3: inserisce cast !llvm.ptr → !cir.ptr e rimappa localDeclMap.
  /// Restituisce una guardia RAII che ripristina le mappature originali.
  /// Riusa la stessa classe RemapGuard del DataSharingProcessor.
  OMPDataSharingProcessor::RemapGuard applyRemapping();

  /// Restituisce true se ci sono variabili di riduzione da gestire.
  bool hasReductionVars() const { return !entries.empty(); }

private:
  CIRGenFunction &cgf;                           // Generatore di codice per la funzione
  CIRGenBuilderTy &builder;                      // Builder MLIR
  mlir::Location loc;                            // Source location
  llvm::SmallVector<OMPReductionVarEntry> entries; // Entry raccolte (una per var di riduzione)

  /// Converte un tipo CIR nel tipo standard MLIR corrispondente.
  /// Stessa logica di OMPDataSharingProcessor::convertCIRTypeToStdType.
  mlir::Type convertCIRTypeToStdType(mlir::Type cirType);

  /// Crea (o riusa) un'op omp.declare_reduction a livello di modulo.
  /// L'op contiene la regione initializer (elemento neutro) e la
  /// regione combiner (operazione di riduzione).
  void getOrCreateDeclareReduction(llvm::StringRef name, mlir::Type stdType,
                                   OMPReductionKind redKind);

  /// Restituisce l'elemento neutro per un dato tipo e operatore:
  /// 0 per add/or/xor, 1 per mul/and. Per i float, 0.0 o 1.0.
  mlir::Value getReductionInitValue(mlir::Type stdType,
                                    OMPReductionKind redKind);

  /// Crea l'operazione di combinazione: add/mul/and/or/xor,
  /// con varianti intere (AddOp, MulOp...) e float (FAddOp, FMulOp...).
  mlir::Value createCombiner(mlir::Value lhs, mlir::Value rhs,
                             mlir::Type stdType, OMPReductionKind redKind);
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
