//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP data sharing (privatization) support for CIR codegen.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPRuntime.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
// RemapGuard
//===----------------------------------------------------------------------===//

OMPDataSharingProcessor::RemapGuard::RemapGuard(
    CIRGenFunction &cgf,
    llvm::SmallVector<std::pair<const VarDecl *, Address>> saved)
    : cgf(cgf), savedAddrs(std::move(saved)) {}

OMPDataSharingProcessor::RemapGuard::~RemapGuard() {
  for (auto &[vd, addr] : savedAddrs)
    cgf.replaceAddrOfLocalVar(vd, addr);
}

OMPDataSharingProcessor::RemapGuard::RemapGuard(RemapGuard &&other) noexcept
    : cgf(other.cgf), savedAddrs(std::move(other.savedAddrs)) {}

//===----------------------------------------------------------------------===//
// OMPDataSharingProcessor
//===----------------------------------------------------------------------===//

OMPDataSharingProcessor::OMPDataSharingProcessor(CIRGenFunction &cgf,
                                                 CIRGenBuilderTy &builder,
                                                 mlir::Location loc)
    : cgf(cgf), builder(builder), loc(loc) {}

mlir::Type
OMPDataSharingProcessor::convertCIRTypeToStdType(mlir::Type cirType) {
  mlir::MLIRContext *ctx = builder.getContext();

  // Integer types (signed and unsigned, all widths).
  if (auto intTy = mlir::dyn_cast<cir::IntType>(cirType))
    return mlir::IntegerType::get(ctx, intTy.getWidth());

  // Bool type.
  if (mlir::isa<cir::BoolType>(cirType))
    return mlir::IntegerType::get(ctx, 1);

  // Float types.
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

  // Pointer types.
  if (mlir::isa<cir::PointerType>(cirType))
    return mlir::LLVM::LLVMPointerType::get(ctx);

  // Unsupported type — emit graceful diagnostic instead of crashing.
  cgf.getCIRGenModule().errorNYI(loc, "private clause for unsupported type");
  return {};
}

void OMPDataSharingProcessor::getOrCreatePrivateOp(
    llvm::StringRef name, mlir::Type stdType,
    mlir::omp::DataSharingClauseType dsType) {
  auto moduleOp = cgf.getCIRGenModule().getModule();
  if (moduleOp.lookupSymbol<mlir::omp::PrivateClauseOp>(name))
    return;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto privateOp =
      mlir::omp::PrivateClauseOp::create(builder, loc, name, stdType, dsType);

  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());

  // Populate the init region. For scalar types, the private variable needs
  // no special initialization — just yield the allocated variable.
  // This mirrors Flang's initTrivialType() in PrivateReductionUtils.cpp.
  //
  // The init region has two block arguments of !llvm.ptr type:
  //   %arg0 = mold (the original host variable, read-only)
  //   %arg1 = allocated private variable
  {
    mlir::Region &initRegion = privateOp.getInitRegion();
    mlir::Block *initBlock = builder.createBlock(
        &initRegion, /*insertPt=*/{}, {llvmPtrTy, llvmPtrTy}, {loc, loc});
    builder.setInsertionPointToEnd(initBlock);
    mlir::omp::YieldOp::create(builder, loc,
                                mlir::ValueRange{initBlock->getArgument(1)});
  }

  // Populate the copy region for firstprivate. The copy region loads the
  // original value from %arg0 and stores it into the private copy %arg1,
  // then yields %arg1. This mirrors Flang's copyFirstPrivateSymbol().
  //
  // The copy region has two block arguments of !llvm.ptr type:
  //   %arg0 = original host variable (source)
  //   %arg1 = allocated private variable (destination)
  if (dsType == mlir::omp::DataSharingClauseType::FirstPrivate) {
    mlir::Region &copyRegion = privateOp.getCopyRegion();
    mlir::Block *copyBlock = builder.createBlock(
        &copyRegion, /*insertPt=*/{}, {llvmPtrTy, llvmPtrTy}, {loc, loc});
    builder.setInsertionPointToEnd(copyBlock);
    mlir::Value origPtr = copyBlock->getArgument(0);
    mlir::Value privPtr = copyBlock->getArgument(1);
    mlir::Value val =
        mlir::LLVM::LoadOp::create(builder, loc, stdType, origPtr);
    mlir::LLVM::StoreOp::create(builder, loc, val, privPtr);
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{privPtr});
  }
}

void OMPDataSharingProcessor::processStep1(
    llvm::ArrayRef<const OMPClause *> clauses,
    OMPPrivateClauseOps &clauseOps, mlir::Operation *insertBeforeOp) {
  // Helper: process a single variable from a private/firstprivate clause.
  auto processVar = [&](const Expr *varExpr,
                        mlir::omp::DataSharingClauseType dsType) {
    const auto *dre = cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
    const auto *vd = cast<VarDecl>(dre->getDecl());

    Address addr = cgf.getAddrOfLocalVar(vd);
    mlir::Value originalAddr = addr.getPointer();
    mlir::Type elementType = addr.getElementType();

    mlir::Type stdType = convertCIRTypeToStdType(elementType);
    if (!stdType)
      return; // errorNYI already emitted

    std::string privatizerName = vd->getNameAsString() + ".privatizer";
    getOrCreatePrivateOp(privatizerName, stdType, dsType);

    entries.push_back({vd, originalAddr, elementType, privatizerName, {}});
  };

  // Collect variables from private and firstprivate clauses.
  for (const OMPClause *c : clauses) {
    if (const auto *privClause = dyn_cast<OMPPrivateClause>(c)) {
      for (const Expr *varExpr : privClause->varlist())
        processVar(varExpr, mlir::omp::DataSharingClauseType::Private);
    } else if (const auto *fpClause = dyn_cast<OMPFirstprivateClause>(c)) {
      for (const Expr *varExpr : fpClause->varlist())
        processVar(varExpr, mlir::omp::DataSharingClauseType::FirstPrivate);
    }
  }

  // Build clauseOps: cast !cir.ptr → !llvm.ptr BEFORE the target op.
  if (!entries.empty()) {
    mlir::Type llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertBeforeOp);
    for (auto &entry : entries) {
      mlir::Value stdPtr =
          mlir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy,
                                                   entry.originalAddr)
              .getResult(0);
      clauseOps.privateVars.push_back(stdPtr);
      clauseOps.privateSyms.push_back(mlir::FlatSymbolRefAttr::get(
          builder.getContext(), entry.privatizerName));
    }
  }
}

void OMPDataSharingProcessor::addBlockArgs(mlir::Block &block) {
  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto &entry : entries)
    entry.blockArg = block.addArgument(llvmPtrTy, loc);
}

OMPDataSharingProcessor::RemapGuard
OMPDataSharingProcessor::applyRemapping() {
  llvm::SmallVector<std::pair<const VarDecl *, Address>> saved;
  for (auto &entry : entries) {
    mlir::Value cirPtr =
        mlir::UnrealizedConversionCastOp::create(
            builder, loc, entry.originalAddr.getType(), entry.blockArg)
            .getResult(0);
    saved.push_back({entry.varDecl, cgf.getAddrOfLocalVar(entry.varDecl)});
    cgf.replaceAddrOfLocalVar(
        entry.varDecl,
        Address(cirPtr, entry.elementType, CharUnits::One()));
  }
  return RemapGuard(cgf, std::move(saved));
}

//===----------------------------------------------------------------------===//
// OMPReductionProcessor
//===----------------------------------------------------------------------===//

OMPReductionProcessor::OMPReductionProcessor(CIRGenFunction &cgf,
                                             CIRGenBuilderTy &builder,
                                             mlir::Location loc)
    : cgf(cgf), builder(builder), loc(loc) {}

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

mlir::Value OMPReductionProcessor::getReductionInitValue(
    mlir::Type stdType, OMPReductionKind redKind) {
  if (mlir::isa<mlir::IntegerType>(stdType)) {
    int64_t initVal = 0;
    switch (redKind) {
    case OMPReductionKind::Add:
    case OMPReductionKind::BitwiseOr:
    case OMPReductionKind::BitwiseXor:
    case OMPReductionKind::LogicalOr:
      initVal = 0;
      break;
    case OMPReductionKind::Multiply:
    case OMPReductionKind::BitwiseAnd:
    case OMPReductionKind::LogicalAnd:
      initVal = 1;
      break;
    }
    return mlir::LLVM::ConstantOp::create(
        builder, loc, stdType,
        builder.getIntegerAttr(stdType, initVal));
  }

  if (mlir::isa<mlir::FloatType>(stdType)) {
    double initVal = 0.0;
    switch (redKind) {
    case OMPReductionKind::Add:
      initVal = 0.0;
      break;
    case OMPReductionKind::Multiply:
      initVal = 1.0;
      break;
    default:
      cgf.getCIRGenModule().errorNYI(
          loc, "reduction init value for non-arithmetic float operator");
      return {};
    }
    return mlir::LLVM::ConstantOp::create(
        builder, loc, stdType,
        builder.getFloatAttr(stdType, initVal));
  }

  cgf.getCIRGenModule().errorNYI(loc, "reduction init for unsupported type");
  return {};
}

mlir::Value OMPReductionProcessor::createCombiner(mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  mlir::Type stdType,
                                                  OMPReductionKind redKind) {
  bool isInt = mlir::isa<mlir::IntegerType>(stdType);
  bool isFloat = mlir::isa<mlir::FloatType>(stdType);

  switch (redKind) {
  case OMPReductionKind::Add:
    if (isInt)
      return mlir::LLVM::AddOp::create(builder, loc, lhs, rhs);
    if (isFloat)
      return mlir::LLVM::FAddOp::create(builder, loc, lhs, rhs);
    break;
  case OMPReductionKind::Multiply:
    if (isInt)
      return mlir::LLVM::MulOp::create(builder, loc, lhs, rhs);
    if (isFloat)
      return mlir::LLVM::FMulOp::create(builder, loc, lhs, rhs);
    break;
  case OMPReductionKind::BitwiseAnd:
    assert(isInt && "bitwise AND requires integer type");
    return mlir::LLVM::AndOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::BitwiseOr:
    assert(isInt && "bitwise OR requires integer type");
    return mlir::LLVM::OrOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::BitwiseXor:
    assert(isInt && "bitwise XOR requires integer type");
    return mlir::LLVM::XOrOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::LogicalAnd:
    assert(isInt && "logical AND requires integer type");
    return mlir::LLVM::AndOp::create(builder, loc, lhs, rhs);
  case OMPReductionKind::LogicalOr:
    assert(isInt && "logical OR requires integer type");
    return mlir::LLVM::OrOp::create(builder, loc, lhs, rhs);
  }

  cgf.getCIRGenModule().errorNYI(loc, "reduction combiner for type/op combo");
  return {};
}

void OMPReductionProcessor::getOrCreateDeclareReduction(
    llvm::StringRef name, mlir::Type stdType, OMPReductionKind redKind) {
  auto moduleOp = cgf.getCIRGenModule().getModule();
  if (moduleOp.lookupSymbol<mlir::omp::DeclareReductionOp>(name))
    return;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto declOp = mlir::omp::DeclareReductionOp::create(
      builder, loc, name, stdType, /*byref_element_type=*/{});

  // Init region: 1 block arg of stdType, yields the neutral element.
  {
    mlir::Region &initRegion = declOp.getInitializerRegion();
    mlir::Block *initBlock =
        builder.createBlock(&initRegion, initRegion.end(), {stdType}, {loc});
    builder.setInsertionPointToEnd(initBlock);
    mlir::Value initVal = getReductionInitValue(stdType, redKind);
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{initVal});
  }

  // Combiner region: 2 block args of stdType, yields the combined result.
  {
    mlir::Region &combinerRegion = declOp.getReductionRegion();
    mlir::Block *combBlock = builder.createBlock(
        &combinerRegion, combinerRegion.end(), {stdType, stdType}, {loc, loc});
    builder.setInsertionPointToEnd(combBlock);
    mlir::Value combined = createCombiner(
        combBlock->getArgument(0), combBlock->getArgument(1), stdType,
        redKind);
    mlir::omp::YieldOp::create(builder, loc, mlir::ValueRange{combined});
  }
}

/// Map Clang's overloaded operator kind to our OMPReductionKind.
static std::optional<OMPReductionKind>
mapOverloadedOpToReductionKind(OverloadedOperatorKind op) {
  switch (op) {
  case OO_Plus:
  case OO_Minus: // reduction(-:x) has same combiner as +
    return OMPReductionKind::Add;
  case OO_Star:
    return OMPReductionKind::Multiply;
  case OO_Amp:
    return OMPReductionKind::BitwiseAnd;
  case OO_Pipe:
    return OMPReductionKind::BitwiseOr;
  case OO_Caret:
    return OMPReductionKind::BitwiseXor;
  case OO_AmpAmp:
    return OMPReductionKind::LogicalAnd;
  case OO_PipePipe:
    return OMPReductionKind::LogicalOr;
  default:
    return std::nullopt;
  }
}

/// Get a human-readable name for a reduction kind.
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

void OMPReductionProcessor::processReductionVars(
    llvm::ArrayRef<const OMPClause *> clauses,
    OMPReductionClauseOps &clauseOps, mlir::Operation *insertBeforeOp) {

  for (const OMPClause *c : clauses) {
    const auto *redClause = dyn_cast<OMPReductionClause>(c);
    if (!redClause)
      continue;

    // Determine the reduction operator kind.
    DeclarationName redName = redClause->getNameInfo().getName();
    OverloadedOperatorKind ooKind = redName.getCXXOverloadedOperator();
    auto redKind = mapOverloadedOpToReductionKind(ooKind);
    if (!redKind) {
      cgf.getCIRGenModule().errorNYI(
          redClause->getBeginLoc(),
          "reduction clause with unsupported operator");
      continue;
    }

    for (const Expr *varExpr : redClause->varlist()) {
      const auto *dre = cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
      const auto *vd = cast<VarDecl>(dre->getDecl());

      Address addr = cgf.getAddrOfLocalVar(vd);
      mlir::Value originalAddr = addr.getPointer();
      mlir::Type elementType = addr.getElementType();

      mlir::Type stdType = convertCIRTypeToStdType(elementType);
      if (!stdType)
        continue;

      // Build a unique name for the declare_reduction op.
      std::string declName =
          (getReductionKindName(*redKind) + "_" + vd->getNameAsString())
              .str();
      getOrCreateDeclareReduction(declName, stdType, *redKind);

      entries.push_back({vd, originalAddr, elementType, declName, {}});
    }
  }

  // Build clauseOps: cast !cir.ptr → !llvm.ptr BEFORE the target op.
  if (!entries.empty()) {
    mlir::Type llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertBeforeOp);
    for (auto &entry : entries) {
      mlir::Value stdPtr =
          mlir::UnrealizedConversionCastOp::create(builder, loc, llvmPtrTy,
                                                   entry.originalAddr)
              .getResult(0);
      clauseOps.reductionVars.push_back(stdPtr);
      clauseOps.reductionSyms.push_back(mlir::FlatSymbolRefAttr::get(
          builder.getContext(), entry.reductionName));
      clauseOps.reductionByref.push_back(false);
    }
  }
}

void OMPReductionProcessor::addBlockArgs(mlir::Block &block) {
  mlir::Type llvmPtrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());
  for (auto &entry : entries)
    entry.blockArg = block.addArgument(llvmPtrTy, loc);
}

OMPDataSharingProcessor::RemapGuard
OMPReductionProcessor::applyRemapping() {
  llvm::SmallVector<std::pair<const VarDecl *, Address>> saved;
  for (auto &entry : entries) {
    mlir::Value cirPtr =
        mlir::UnrealizedConversionCastOp::create(
            builder, loc, entry.originalAddr.getType(), entry.blockArg)
            .getResult(0);
    saved.push_back({entry.varDecl, cgf.getAddrOfLocalVar(entry.varDecl)});
    cgf.replaceAddrOfLocalVar(
        entry.varDecl,
        Address(cirPtr, entry.elementType, CharUnits::One()));
  }
  return OMPDataSharingProcessor::RemapGuard(cgf, std::move(saved));
}
