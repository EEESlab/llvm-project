//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP clause nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
template <typename OpTy>
class OpenMPClauseCIREmitter final
    : public ConstOMPClauseVisitor<OpenMPClauseCIREmitter<OpTy>> {
  OpTy &operation;
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

public:
  OpenMPClauseCIREmitter(OpTy &operation, CIRGen::CIRGenFunction &cgf,
                         CIRGen::CIRGenBuilderTy &builder)
      : operation(operation), cgf(cgf), builder(builder) {}

  void VisitOMPClause(const OMPClause *clause) {
    cgf.cgm.errorNYI(clause->getBeginLoc(), "OpenMPClause ",
                     llvm::omp::getOpenMPClauseName(clause->getClauseKind()));
  }

  void VisitOMPProcBindClause(const OMPProcBindClause *clause) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::ParallelOp>) {
      mlir::omp::ClauseProcBindKind kind;
      switch (clause->getProcBindKind()) {
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_master:
        kind = mlir::omp::ClauseProcBindKind::Master;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_close:
        kind = mlir::omp::ClauseProcBindKind::Close;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_spread:
        kind = mlir::omp::ClauseProcBindKind::Spread;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_primary:
        kind = mlir::omp::ClauseProcBindKind::Primary;
        break;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_default:
        // 'default' in the classic-codegen does no runtime call/doesn't
        // really do anything. So this is a no-op, and thus shouldn't change
        // the IR.
        return;
      case llvm::omp::ProcBindKind::OMP_PROC_BIND_unknown:
        llvm_unreachable("unknown proc-bind kind");
      }
      operation.setProcBindKind(kind);
    } else {
      cgf.cgm.errorNYI(
          clause->getBeginLoc(),
          "OMPProcBindClause unimplemented on this directive kind");
    }
  }

  // Private and firstprivate clauses are handled by OMPDataSharingProcessor
  // in the directive emitters. These visitors intentionally do nothing.
  void VisitOMPPrivateClause(const OMPPrivateClause *) {}
  void VisitOMPFirstprivateClause(const OMPFirstprivateClause *) {}

  // Reduction clauses are handled by OMPReductionProcessor in the directive
  // emitters. This visitor intentionally does nothing.
  void VisitOMPReductionClause(const OMPReductionClause *) {}

  void VisitOMPScheduleClause(const OMPScheduleClause *clause) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::WsloopOp>) {
      mlir::MLIRContext *ctx = builder.getContext();

      // Map Clang schedule kind to MLIR schedule kind.
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
      operation.setScheduleKindAttr(
          mlir::omp::ClauseScheduleKindAttr::get(ctx, schedKind));

      // Map schedule modifiers (monotonic/nonmonotonic/simd).
      auto mapModifier = [&](OpenMPScheduleClauseModifier mod) {
        switch (mod) {
        case OMPC_SCHEDULE_MODIFIER_monotonic:
          operation.setScheduleModAttr(mlir::omp::ScheduleModifierAttr::get(
              ctx, mlir::omp::ScheduleModifier::monotonic));
          break;
        case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
          operation.setScheduleModAttr(mlir::omp::ScheduleModifierAttr::get(
              ctx, mlir::omp::ScheduleModifier::nonmonotonic));
          break;
        case OMPC_SCHEDULE_MODIFIER_simd:
          operation.setScheduleSimdAttr(builder.getUnitAttr());
          break;
        default:
          break;
        }
      };
      mapModifier(clause->getFirstScheduleModifier());
      mapModifier(clause->getSecondScheduleModifier());

      // Handle chunk size expression.
      if (const Expr *chunkExpr = clause->getChunkSize()) {
        mlir::Value cirChunk = cgf.emitScalarExpr(chunkExpr);
        // Convert CIR int to standard MLIR int for the omp dialect.
        if (auto cirIntTy = mlir::dyn_cast<cir::IntType>(cirChunk.getType())) {
          mlir::Type stdIntTy = builder.getIntegerType(cirIntTy.getWidth());
          mlir::Value stdChunk =
              mlir::UnrealizedConversionCastOp::create(builder,
                  cgf.getLoc(clause->getBeginLoc()), stdIntTy, cirChunk)
                  .getResult(0);
          operation.getScheduleChunkMutable().assign(stdChunk);
        }
      }
    } else {
      cgf.cgm.errorNYI(
          clause->getBeginLoc(),
          "OMPScheduleClause unimplemented on this directive kind");
    }
  }

  void emitClauses(ArrayRef<const OMPClause *> clauses) {
    for (const auto *c : clauses)
      this->Visit(c);
  }
};
template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder) {
  return OpenMPClauseCIREmitter<OpTy>(op, cgf, builder);
}
} // namespace

template <typename Op>
void CIRGenFunction::emitOpenMPClauses(Op &op,
                                       ArrayRef<const OMPClause *> clauses) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPoint(op);
  makeClauseEmitter(op, *this, builder).emitClauses(clauses);
}

// We're defining the template for this in a .cpp file, so we have to explicitly
// specialize the templates.
#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenMPClauses<N>(                          \
      N &, ArrayRef<const OMPClause *>);
EXPL_SPEC(mlir::omp::ParallelOp)
EXPL_SPEC(mlir::omp::WsloopOp)
#undef EXPL_SPEC
