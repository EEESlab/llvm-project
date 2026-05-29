//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emissione delle clausole OpenMP come codice CIR.
//
// Questo file implementa il pattern Visitor per le clausole OpenMP.
// Ogni clausola del #pragma omp viene visitata e tradotta negli
// attributi/operandi dell'op MLIR corrispondente.
//
// NOTA IMPORTANTE: alcune clausole sono gestite altrove:
//   - private/firstprivate → OMPDataSharingProcessor (CIRGenOpenMPRuntime)
//   - reduction → OMPReductionProcessor (CIRGenOpenMPRuntime)
//   - collapse → consumata direttamente dagli emitter di direttiva
//   - shared → no-op (le variabili sono shared per default in OpenMP)
//
// Per queste clausole, i visitor qui sono intenzionalmente vuoti (no-op)
// per evitare che il visitor di default emetta un errore "Not Yet Implemented".
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

/// Visitor template per le clausole OpenMP.
///
/// È parametrizzato sul tipo dell'op MLIR (OpTy) a cui le clausole
/// appartengono. Questo permette di usare `if constexpr` per gestire
/// clausole che hanno significato diverso a seconda dell'op
/// (es. proc_bind ha senso solo su ParallelOp, schedule solo su WsloopOp).
///
/// Eredita da ConstOMPClauseVisitor, che fornisce il dispatch automatico
/// dal tipo della clausola al metodo Visit* corrispondente.
template <typename OpTy>
class OpenMPClauseCIREmitter final
    : public ConstOMPClauseVisitor<OpenMPClauseCIREmitter<OpTy>> {
  OpTy &operation;                 // L'op MLIR su cui impostare gli attributi
  CIRGen::CIRGenFunction &cgf;    // Generatore di codice per la funzione
  CIRGen::CIRGenBuilderTy &builder; // Builder MLIR per creare op/attributi

public:
  OpenMPClauseCIREmitter(OpTy &operation, CIRGen::CIRGenFunction &cgf,
                         CIRGen::CIRGenBuilderTy &builder)
      : operation(operation), cgf(cgf), builder(builder) {}

  /// Visitor di default: viene invocato per qualsiasi clausola che non ha
  /// un metodo Visit* specifico. Emette un errore "Not Yet Implemented"
  /// con il nome della clausola non gestita.
  void VisitOMPClause(const OMPClause *clause) {
    cgf.cgm.errorNYI(clause->getBeginLoc(), "OpenMPClause ",
                     llvm::omp::getOpenMPClauseName(clause->getClauseKind()));
  }

  /// Clausola proc_bind(master|close|spread|primary).
  ///
  /// Specifica come i thread vengono assegnati ai processori.
  /// Ha significato solo su omp.parallel — per altre op (es. WsloopOp
  /// quando si è dentro un `parallel for`) viene ignorata perché
  /// sarà gestita dall'op parallel esterna.
  void VisitOMPProcBindClause(const OMPProcBindClause *clause) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::ParallelOp>) {
      mlir::omp::ClauseProcBindKind kind;
      switch (clause->getProcBindKind()) {
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_master:
        // I thread si eseguono sullo stesso processore del master thread
        kind = mlir::omp::ClauseProcBindKind::Master;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_close:
        // I thread si eseguono su processori "vicini" al master
        kind = mlir::omp::ClauseProcBindKind::Close;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_spread:
        // I thread si distribuiscono uniformemente sui processori
        kind = mlir::omp::ClauseProcBindKind::Spread;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_primary:
        // Come master, ma è il nome moderno (OpenMP 5.1+)
        kind = mlir::omp::ClauseProcBindKind::Primary;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_default:
        // Il valore 'default' non genera chiamate al runtime — è un no-op.
        return;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_unknown:
        llvm_unreachable("unknown proc-bind kind");
      }
      // Imposta l'attributo proc_bind_kind sull'op ParallelOp.
      operation.setProcBindKind(kind);
    }
    // Per op non-ParallelOp (es. WsloopOp in `parallel for`), proc_bind è
    // gestito quando si processa l'op parallel esterna. No-op intenzionale.
  }

  // === Clausole gestite altrove — visitor vuoti ===

  // Le clausole private e firstprivate sono gestite da OMPDataSharingProcessor
  // negli emitter di direttiva (emitOMPParallelDirective, ecc.).
  // Questi visitor vuoti impediscono al default VisitOMPClause di
  // emettere un errore NYI.
  void VisitOMPPrivateClause(const OMPPrivateClause *) {}
  void VisitOMPFirstprivateClause(const OMPFirstprivateClause *) {}

  // La clausola reduction è gestita da OMPReductionProcessor.
  void VisitOMPReductionClause(const OMPReductionClause *) {}

  // La clausola collapse è consumata direttamente dagli emitter di
  // direttiva tramite getLoopsNumber(). Non serve tradurla in IR.
  void VisitOMPCollapseClause(const OMPCollapseClause *) {}

  // La clausola shared è un no-op: in OpenMP le variabili sono shared
  // per default, quindi non serve generare codice aggiuntivo.
  void VisitOMPSharedClause(const OMPSharedClause *) {}

  /// Clausola nowait.
  ///
  /// Elimina la barriera implicita alla fine della regione di
  /// worksharing, permettendo ai thread di proseguire senza
  /// sincronizzarsi. È rappresentata come un UnitAttr (presente =
  /// nowait, assente = barriera implicita).
  ///
  /// Ha significato solo su omp.wsloop (#pragma omp for nowait). Per
  /// omp.single la clausola è gestita direttamente nel suo emitter di
  /// direttiva. Per altre op (es. ParallelOp in `parallel for`) la
  /// clausola non è valida e viene ignorata qui.
  void VisitOMPNowaitClause(const OMPNowaitClause *) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::WsloopOp>)
      operation.setNowaitAttr(builder.getUnitAttr());
  }

  /// Clausola schedule(kind[, chunk_size]).
  ///
  /// Specifica come le iterazioni di un loop vengono distribuite tra
  /// i thread. Ha significato solo su omp.wsloop.
  ///
  /// Tipi di schedule supportati:
  ///   - static:  iterazioni divise in blocchi uguali assegnati staticamente
  ///   - dynamic: ogni thread prende chunk_size iterazioni alla volta
  ///   - guided:  come dynamic, ma il chunk si riduce progressivamente
  ///   - auto:    il compilatore/runtime decide la strategia migliore
  ///   - runtime: la strategia è scelta a runtime via OMP_SCHEDULE
  ///
  /// Modificatori opzionali:
  ///   - monotonic:    le iterazioni sono assegnate in ordine crescente
  ///   - nonmonotonic: nessuna garanzia sull'ordine
  ///   - simd:         allinea il chunk alla larghezza SIMD
  void VisitOMPScheduleClause(const OMPScheduleClause *clause) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::WsloopOp>) {
      mlir::MLIRContext *ctx = builder.getContext();

      // Mappa il tipo di schedule Clang → tipo di schedule MLIR OMP.
      mlir::omp::ClauseScheduleKind schedKind;
      switch (clause->getScheduleKind()) {
      case OMPC_SCHEDULE_static:
        schedKind = mlir::omp::ClauseScheduleKind::Static;
        break;
      case OMPC_SCHEDULE_dynamic:
        schedKind = mlir::omp::ClauseScheduleKind::Dynamic;
        break;
      case OMPC_SCHEDULE_guided:
        schedKind = mlir::omp::ClauseScheduleKind::Guided;
        break;
      case OMPC_SCHEDULE_auto:
        schedKind = mlir::omp::ClauseScheduleKind::Auto;
        break;
      case OMPC_SCHEDULE_runtime:
        schedKind = mlir::omp::ClauseScheduleKind::Runtime;
        break;
      case OMPC_SCHEDULE_unknown:
        llvm_unreachable("unknown schedule kind");
      }
      // Imposta l'attributo schedule_kind sull'op WsloopOp.
      operation.setScheduleKindAttr(
          mlir::omp::ClauseScheduleKindAttr::get(ctx, schedKind));

      // Gestione dei modificatori di schedule (monotonic/nonmonotonic/simd).
      // Una clausola schedule può avere fino a 2 modificatori.
      auto mapModifier = [&](OpenMPScheduleClauseModifier mod) {
        switch (mod) {
        case OMPC_SCHEDULE_MODIFIER_monotonic:
          // Imposta il modificatore monotonic sull'attributo schedule_mod
          operation.setScheduleModAttr(mlir::omp::ScheduleModifierAttr::get(
              ctx, mlir::omp::ScheduleModifier::monotonic));
          break;
        case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
          // Imposta il modificatore nonmonotonic
          operation.setScheduleModAttr(mlir::omp::ScheduleModifierAttr::get(
              ctx, mlir::omp::ScheduleModifier::nonmonotonic));
          break;
        case OMPC_SCHEDULE_MODIFIER_simd:
          // Il modificatore simd usa un attributo UnitAttr separato
          operation.setScheduleSimdAttr(builder.getUnitAttr());
          break;
        default:
          // Nessun modificatore o modificatore sconosciuto — ignorato
          break;
        }
      };
      // Applica il primo e il secondo modificatore (se presenti)
      mapModifier(clause->getFirstScheduleModifier());
      mapModifier(clause->getSecondScheduleModifier());

      // Gestione del chunk size (es. schedule(dynamic, 4)).
      // Se presente un'espressione per il chunk size, la valuta e
      // la converte da tipo CIR intero a tipo intero standard MLIR.
      if (const Expr *chunkExpr = clause->getChunkSize()) {
        // Emette il codice per valutare l'espressione del chunk size
        mlir::Value cirChunk = cgf.emitScalarExpr(chunkExpr);
        // Converte il tipo CIR int → tipo standard MLIR int,
        // necessario perché il dialetto OMP non conosce i tipi CIR.
        if (auto cirIntTy = mlir::dyn_cast<cir::IntType>(cirChunk.getType())) {
          mlir::Type stdIntTy = builder.getIntegerType(cirIntTy.getWidth());
          mlir::Value stdChunk =
              mlir::UnrealizedConversionCastOp::create(builder,
                  cgf.getLoc(clause->getBeginLoc()), stdIntTy, cirChunk)
                  .getResult(0);
          // Imposta il chunk size come operando dell'op WsloopOp
          operation.getScheduleChunkMutable().assign(stdChunk);
        }
      }
    }
    // Per op non-WsloopOp (es. ParallelOp in `parallel for`), schedule è
    // gestito quando si processa l'op wsloop interna. No-op intenzionale.
  }

  /// Punto d'ingresso: visita tutte le clausole una per una.
  void emitClauses(ArrayRef<const OMPClause *> clauses) {
    for (const auto *c : clauses)
      this->Visit(c); // Dispatch al Visit* corretto tramite il visitor pattern
  }
};

/// Helper per creare un OpenMPClauseCIREmitter con deduzione automatica
/// del tipo template OpTy.
template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder) {
  return OpenMPClauseCIREmitter<OpTy>(op, cgf, builder);
}
} // namespace

/// Funzione template chiamata dagli emitter di direttiva per processare
/// tutte le clausole non gestite altrove.
///
/// Imposta il punto di inserimento prima dell'op (per rispettare la
/// dominanza SSA nel caso servisse generare operazioni) e poi
/// invoca il visitor su tutte le clausole.
///
/// Questa è una funzione template perché il tipo dell'op varia
/// (ParallelOp, WsloopOp, ecc.) e il visitor ha bisogno del tipo
/// concreto per usare if constexpr.
template <typename Op>
void CIRGenFunction::emitOpenMPClauses(Op &op,
                                       ArrayRef<const OMPClause *> clauses) {
  // Salva e ripristina il punto di inserimento del builder
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  // Si inserisce prima dell'op per la dominanza SSA
  builder.setInsertionPoint(op);
  // Crea il visitor e processa tutte le clausole
  makeClauseEmitter(op, *this, builder).emitClauses(clauses);
}

// Le istanziazioni esplicite del template sono necessarie perché
// l'implementazione è in un file .cpp (non in un header).
// Senza queste, il linker non troverebbe il codice per i tipi concreti.
#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenMPClauses<N>(                          \
      N &, ArrayRef<const OMPClause *>);
EXPL_SPEC(mlir::omp::ParallelOp) // Per #pragma omp parallel
EXPL_SPEC(mlir::omp::WsloopOp)   // Per #pragma omp for
#undef EXPL_SPEC
