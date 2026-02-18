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

  void VisitOMPPrivateClause(const OMPPrivateClause *clause) {
    if constexpr (std::is_same_v<OpTy, mlir::omp::WsloopOp>) {
      for (const Expr *varExpr : clause->varlist()) {
        const auto *dre =
            cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
        const auto *vd = cast<VarDecl>(dre->getDecl());

        Address addr = cgf.getAddrOfLocalVar(vd);
        mlir::Value originalAddr = addr.getPointer();
        mlir::Type elementType = addr.getElementType();

        // Convert CIR element type to standard MLIR type for omp.private.
        mlir::Type stdType;
        if (auto cirIntTy = mlir::dyn_cast<cir::IntType>(elementType))
          stdType = builder.getIntegerType(cirIntTy.getWidth());
        else
          llvm_unreachable("unsupported private variable type");

        // Build a unique privatizer name.
        std::string privatizerName =
            vd->getNameAsString() + ".privatizer";

        // Reuse an existing privatizer if one with the same name exists.
        auto moduleOp = cgf.getCIRGenModule().getModule();
        if (!moduleOp.lookupSymbol<mlir::omp::PrivateClauseOp>(
                privatizerName)) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(moduleOp.getBody());
          mlir::omp::PrivateClauseOp::create(
              builder, operation.getLoc(), privatizerName,
              mlir::TypeAttr::get(stdType),
              mlir::omp::DataSharingClauseType::Private);
        }

        cgf.currentOMPPrivateVars.push_back(
            {vd, originalAddr, elementType, privatizerName});
      }
    } else {
      cgf.cgm.errorNYI(clause->getBeginLoc(),
                       "OMPPrivateClause on this directive kind");
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
