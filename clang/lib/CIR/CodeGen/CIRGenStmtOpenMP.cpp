//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace clang;
using namespace clang::CIRGen;

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
mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::LogicalResult res = mlir::success();
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  auto parallelOp =
      mlir::omp::ParallelOp::create(builder, begin, retTy, operands);

  // Process non-private clauses (e.g. proc_bind).
  emitOpenMPClauses(parallelOp, s.clauses());

  // Data sharing: collect private vars, create omp.private ops, build operands.
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, parallelOp);

  if (dsp.hasPrivateVars()) {
    parallelOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    parallelOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // Reduction: collect reduction vars, create omp.declare_reduction ops.
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

  {
    mlir::Block &block = parallelOp.getRegion().emplaceBlock();
    dsp.addBlockArgs(block);
    rdp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    // RAII remapping: casts block args to CIR pointers and remaps localDeclMap.
    auto remapGuard = dsp.applyRemapping();
    auto redRemapGuard = rdp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    if (s.hasCancel())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Cancel");
    if (s.getTaskReductionRefExpr())
      getCIRGenModule().errorNYI(s.getBeginLoc(),
                                 "OpenMP Parallel with Task Reduction");
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
    const Stmt *bodyStmt = cs->getCapturedStmt();
    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    // remapGuard restores original variable mappings on scope exit.
    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}

// Helpers for emitOMPForDirective / emitOMPParallelForDirective, which lower
// loop directives into omp.wsloop + omp.loop_nest.

namespace {
/// Extract integer literal value from an expression, if present.
static std::optional<int64_t> getIntLiteralValue(const Expr *expr) {
  if (const auto *intLit = dyn_cast<IntegerLiteral>(expr->IgnoreImpCasts()))
    return intLit->getValue().getSExtValue();
  return std::nullopt;
}

/// Ensure a CIR value has the given CIR integer type, inserting an integral
/// cast if necessary. Loads through CIR pointers first.
static mlir::Value ensureCIRIntType(CIRGenBuilderTy &builder,
                                    mlir::Location loc, mlir::Value cirValue,
                                    cir::IntType targetCIRType) {
  if (mlir::isa<cir::PointerType>(cirValue.getType()))
    cirValue = cir::LoadOp::create(builder, loc, cirValue).getResult();

  if (cirValue.getType() == targetCIRType)
    return cirValue;

  return builder.createCast(loc, cir::CastKind::integral, cirValue,
                            targetCIRType);
}

/// Convert a CIR integer value to a standard MLIR integer type suitable for
/// use as an omp.loop_nest operand.
static mlir::Value cirIntToStdInt(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value cirValue) {
  auto cirIntType = mlir::cast<cir::IntType>(cirValue.getType());
  mlir::Type stdIntType = builder.getIntegerType(cirIntType.getWidth());
  return mlir::UnrealizedConversionCastOp::create(builder, loc, stdIntType,
                                                  cirValue)
      .getResult(0);
}
} // anonymous namespace

/// Extract the ForStmt from an OpenMP loop directive's CapturedStmt, parse
/// its init/cond/inc to produce loop bounds as CIR values, emit the loop init
/// statement (alloca for IV), and convert bounds to standard MLIR integers.
/// On success, populates `currentOMPLoopBounds`.
mlir::LogicalResult CIRGenFunction::extractOMPLoopBounds(
    const ForStmt *forStmt, mlir::Location loc) {

  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false;
  Address savedAddr = Address::invalid();

  // Extract loop variable type and lower bound.
  // Two forms are supported:
  //   1. DeclStmt:  for (int i = 0; ...)   — variable declared in the init.
  //   2. Expr:      for (i = 0; ...)        — variable declared outside.
  const VarDecl *varDecl = nullptr;
  const Expr *initExpr = nullptr;

  if (const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit())) {
    varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl());
    if (!varDecl || !varDecl->hasInit())
      return mlir::failure();
    initExpr = varDecl->getInit();
  } else if (const auto *binOp = dyn_cast_or_null<BinaryOperator>(
                 forStmt->getInit())) {
    // Handle `i = 0` where i is declared outside the for loop.
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

  // The loop variable's CIR integer type is the canonical type for all bounds.
  QualType loopVarQType = varDecl->getType();
  auto cirType = convertType(loopVarQType);
  auto cirIntType = mlir::cast<cir::IntType>(cirType);

  // Extract lower bound.
  if (auto constVal = getIntLiteralValue(initExpr)) {
    lowerBound = builder.getConstInt(loc, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = emitScalarExpr(initExpr);
    lowerBound = ensureCIRIntType(builder, loc, cirValue, cirIntType);
  }

  // Extract upper bound and comparison operator.
  const auto *condBinOp = dyn_cast_or_null<BinaryOperator>(forStmt->getCond());
  if (!condBinOp)
    return mlir::failure();

  BinaryOperatorKind opKind = condBinOp->getOpcode();

  // Determine which side of the comparison holds the upper bound.
  // Canonical forms: `i < ub`, `i <= ub` (var on LHS, bound on RHS)
  //                  `ub > i`, `ub >= i` (bound on LHS, var on RHS)
  const Expr *boundExpr = nullptr;
  if (opKind == BO_LT || opKind == BO_LE) {
    boundExpr = condBinOp->getRHS();
    inclusive = (opKind == BO_LE);
  } else if (opKind == BO_GT || opKind == BO_GE) {
    boundExpr = condBinOp->getLHS();
    inclusive = (opKind == BO_GE);
  } else {
    return mlir::failure();
  }

  if (auto constVal = getIntLiteralValue(boundExpr)) {
    upperBound = builder.getConstInt(loc, cirIntType, *constVal);
  } else {
    mlir::Value cirValue = emitScalarExpr(boundExpr);
    upperBound = ensureCIRIntType(builder, loc, cirValue, cirIntType);
  }

  // Extract step.
  if (const auto *unaryOp =
          dyn_cast_or_null<UnaryOperator>(forStmt->getInc())) {
    int64_t val = unaryOp->isIncrementOp() ? 1 : -1;
    step = builder.getConstInt(loc, cirIntType, val);
  } else if (const auto *binOp =
                 dyn_cast_or_null<BinaryOperator>(forStmt->getInc())) {
    const Expr *stepExpr = nullptr;

    if (binOp->isCompoundAssignmentOp()) {
      stepExpr = binOp->getRHS();
    } else if (binOp->isAssignmentOp()) {
      // i = i + step or i = step + i
      if (auto *subBinOp =
              dyn_cast<BinaryOperator>(binOp->getRHS()->IgnoreImpCasts())) {
        const Expr *lhs = subBinOp->getLHS()->IgnoreImpCasts();
        const Expr *rhs = subBinOp->getRHS()->IgnoreImpCasts();
        // Identify which operand is the loop variable and which is the step.
        if (auto *lhsRef = dyn_cast<DeclRefExpr>(lhs)) {
          stepExpr = (lhsRef->getDecl() == varDecl) ? rhs : lhs;
        } else if (auto *rhsRef = dyn_cast<DeclRefExpr>(rhs)) {
          stepExpr = (rhsRef->getDecl() == varDecl) ? lhs : rhs;
        }
      }
    }

    if (stepExpr) {
      if (auto constVal = getIntLiteralValue(stepExpr)) {
        step = builder.getConstInt(loc, cirIntType, *constVal);
      } else {
        mlir::Value cirValue = emitScalarExpr(stepExpr);
        step = ensureCIRIntType(builder, loc, cirValue, cirIntType);
      }
    }
  }

  // Default to unit step if not recognized.
  if (!step)
    step = builder.getConstInt(loc, cirIntType, 1);

  // Emit the loop init to create the alloca for the induction variable.
  // For DeclStmt (`int i = 0`), emitting the statement creates the alloca
  // naturally. For assignment (`i = 0`), the variable is declared outside the
  // loop. OpenMP requires the induction variable to be implicitly private, so
  // we create a new private alloca inside the current region and remap
  // localDeclMap to use it.
  if (const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit())) {
    if (emitStmt(declStmt, /*useCurrentScope=*/true).failed())
      return mlir::failure();
  } else if (forStmt->getInit()) {
    // Assignment init (e.g. `i = 0`): create a private alloca for implicit
    // privatization of the induction variable.
    savedAddr = getAddrOfLocalVar(varDecl);
    Address privateAddr =
        createMemTemp(loopVarQType, loc, varDecl->getName() + ".iv");
    cir::StoreOp::create(builder, loc, lowerBound, privateAddr.getPointer(),
                         /*is_volatile=*/nullptr, /*alignment=*/nullptr,
                         /*sync_scope=*/nullptr, /*mem_order=*/nullptr);
    replaceAddrOfLocalVar(varDecl, privateAddr);
  }

  // Convert CIR integer bounds to standard MLIR integers at the boundary.
  // omp.loop_nest requires IntLikeType (AnyInteger | Index), not CIR types.
  mlir::Value stdLB = cirIntToStdInt(builder, loc, lowerBound);
  mlir::Value stdUB = cirIntToStdInt(builder, loc, upperBound);
  mlir::Value stdStep = cirIntToStdInt(builder, loc, step);
  mlir::Type loopBoundsType = stdLB.getType();

  currentOMPLoopBounds =
      LoopBounds{stdLB, stdUB, stdStep, loopBoundsType, varDecl, inclusive,
                 savedAddr};
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &s) {

  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());

  // Extract the underlying canonical `for` loop from the CapturedStmt.
  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (!forStmt)
    return mlir::failure();

  // Extract loop bounds, emit loop init, and populate currentOMPLoopBounds.
  if (extractOMPLoopBounds(forStmt, begin).failed())
    return mlir::failure();

  // Create wsloop and process clauses.
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto wsloopOp = mlir::omp::WsloopOp::create(builder, begin, retTy, operands);

  // Process non-private clauses (schedule, etc.).
  emitOpenMPClauses(wsloopOp, s.clauses());

  // Data sharing: collect private vars, create omp.private ops, build operands.
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, wsloopOp);

  if (dsp.hasPrivateVars()) {
    wsloopOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    wsloopOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // Reduction: collect reduction vars, create omp.declare_reduction ops.
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

  // Create the wsloop region block. Block args for private and reduction vars
  // are added here; the corresponding remapping casts are emitted inside the
  // loop_nest body (in emitForStmt) to satisfy the wsloop "exactly one nested
  // op" constraint.
  mlir::Region &region = wsloopOp.getRegion();
  mlir::Block *block = new mlir::Block();
  region.push_back(block);
  dsp.addBlockArgs(*block);
  rdp.addBlockArgs(*block);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  // Store processors for deferred remapping in emitForStmt.
  assert(!currentOMPDataSharingProcessor &&
         "nested wsloop privatization not supported");
  currentOMPDataSharingProcessor = &dsp;
  currentOMPReductionProcessor = &rdp;

  // Save restore info before emitStmt, which clears currentOMPLoopBounds
  // inside emitForStmt to prevent nested for-loops from being treated as
  // additional omp.loop_nest ops.
  Address savedAddr = currentOMPLoopBounds->savedInductionVarAddr;
  const VarDecl *inductionVar = currentOMPLoopBounds->inductionVar;

  // Emit the ForStmt body (will create loop_nest as the single nested op).
  // Variable remapping for private/reduction vars happens inside the loop_nest
  // body. Note: currentOMPLoopBounds is cleared inside emitForStmt after the
  // loop_nest is created.
  if (emitStmt(forStmt, /*useCurrentScope=*/false).failed())
    res = mlir::failure();

  currentOMPDataSharingProcessor = nullptr;
  currentOMPReductionProcessor = nullptr;

  // Restore the original address mapping for the induction variable if it was
  // implicitly privatized (declared outside the for-init).
  if (savedAddr.isValid())
    replaceAddrOfLocalVar(inductionVar, savedAddr);

  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskwaitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskyieldDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::omp::BarrierOp::create(builder, getLoc(s.getBeginLoc()));
  assert(s.clauses().empty() && "omp barrier doesn't support clauses");
  return mlir::success();
}
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
mlir::LogicalResult
CIRGenFunction::emitOMPSingleDirective(const OMPSingleDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  mlir::omp::SingleOperands clauseOps;

  // Handle nowait clause.
  for (const OMPClause *c : s.clauses()) {
    if (isa<OMPNowaitClause>(c))
      clauseOps.nowait = builder.getUnitAttr();
  }

  auto singleOp =
      mlir::omp::SingleOp::create(builder, begin, clauseOps);

  // Data sharing: collect private/firstprivate vars.
  OMPPrivateClauseOps privClauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), privClauseOps, singleOp);

  if (dsp.hasPrivateVars()) {
    singleOp.getPrivateVarsMutable().append(privClauseOps.privateVars);
    singleOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), privClauseOps.privateSyms));
  }

  {
    mlir::Block &block = singleOp.getRegion().emplaceBlock();
    dsp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);

    auto remapGuard = dsp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    const Stmt *bodyStmt = s.getAssociatedStmt();
    if (const auto *cs = dyn_cast<CapturedStmt>(bodyStmt))
      bodyStmt = cs->getCapturedStmt();
    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}
mlir::LogicalResult
CIRGenFunction::emitOMPMasterDirective(const OMPMasterDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  auto masterOp = mlir::omp::MasterOp::create(builder, begin);

  {
    mlir::Block &block = masterOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    const Stmt *bodyStmt = s.getAssociatedStmt();
    if (const auto *cs = dyn_cast<CapturedStmt>(bodyStmt))
      bodyStmt = cs->getCapturedStmt();
    res = emitStmt(bodyStmt, /*useCurrentScope=*/true);

    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}
mlir::LogicalResult
CIRGenFunction::emitOMPCriticalDirective(const OMPCriticalDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCriticalDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(const OMPParallelForDirective &s) {
  mlir::LogicalResult res = mlir::success();
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // Extract the underlying canonical `for` loop from the CapturedStmt.
  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const ForStmt *forStmt = dyn_cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (!forStmt)
    return mlir::failure();

  if (s.hasCancel())
    getCIRGenModule().errorNYI(s.getBeginLoc(),
                               "OpenMP ParallelFor with Cancel");
  if (s.getTaskReductionRefExpr())
    getCIRGenModule().errorNYI(s.getBeginLoc(),
                               "OpenMP ParallelFor with Task Reduction");

  // --- Create outer omp.parallel ---
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto parallelOp =
      mlir::omp::ParallelOp::create(builder, begin, retTy, operands);

  // Process parallel-level clauses (proc_bind, num_threads, etc.).
  emitOpenMPClauses(parallelOp, s.clauses());

  // Data sharing: private vars go on the parallel op.
  OMPPrivateClauseOps clauseOps;
  OMPDataSharingProcessor dsp(*this, builder, begin);
  dsp.processStep1(s.clauses(), clauseOps, parallelOp);

  if (dsp.hasPrivateVars()) {
    parallelOp.getPrivateVarsMutable().append(clauseOps.privateVars);
    parallelOp.setPrivateSymsAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauseOps.privateSyms));
  }

  // Reduction: reduction vars go on the parallel op.
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

  {
    mlir::Block &block = parallelOp.getRegion().emplaceBlock();
    dsp.addBlockArgs(block);
    rdp.addBlockArgs(block);

    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    // RAII remapping: cast block args to CIR pointers and remap localDeclMap.
    auto remapGuard = dsp.applyRemapping();
    auto redRemapGuard = rdp.applyRemapping();

    LexicalScope ls{*this, begin, builder.getInsertionBlock()};

    // Extract loop bounds and emit loop init inside the parallel region.
    if (extractOMPLoopBounds(forStmt, begin).failed())
      return mlir::failure();

    // --- Create inner omp.wsloop ---
    llvm::SmallVector<mlir::Type> wsRetTy;
    llvm::SmallVector<mlir::Value> wsOperands;
    auto wsloopOp =
        mlir::omp::WsloopOp::create(builder, begin, wsRetTy, wsOperands);

    // Process wsloop-level clauses (schedule, etc.).
    emitOpenMPClauses(wsloopOp, s.clauses());

    // Create the wsloop region block (no private/reduction block args — those
    // are on the parallel op).
    mlir::Region &wsRegion = wsloopOp.getRegion();
    mlir::Block *wsBlock = new mlir::Block();
    wsRegion.push_back(wsBlock);

    // Save restore info before emitStmt, which clears currentOMPLoopBounds.
    Address savedAddr = currentOMPLoopBounds->savedInductionVarAddr;
    const VarDecl *inductionVar = currentOMPLoopBounds->inductionVar;

    {
      mlir::OpBuilder::InsertionGuard wsGuard(builder);
      builder.setInsertionPointToStart(wsBlock);

      // Emit the ForStmt which creates the omp.loop_nest as the single nested
      // op inside wsloop. No deferred remapping needed — private/reduction
      // vars are already remapped at the parallel level.
      // Note: currentOMPLoopBounds is cleared inside emitForStmt after the
      // loop_nest is created.
      if (emitStmt(forStmt, /*useCurrentScope=*/false).failed())
        res = mlir::failure();
    }

    // Restore the original address mapping for the induction variable if it
    // was implicitly privatized (declared outside the for-init).
    if (savedAddr.isValid())
      replaceAddrOfLocalVar(inductionVar, savedAddr);

    // remapGuard restores original variable mappings on scope exit.
    mlir::omp::TerminatorOp::create(builder, end);
  }

  return res;
}
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
mlir::LogicalResult
CIRGenFunction::emitOMPTaskDirective(const OMPTaskDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskDirective");
  return mlir::failure();
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
