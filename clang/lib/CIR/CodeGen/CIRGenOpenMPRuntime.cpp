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

void OMPDataSharingProcessor::getOrCreatePrivateOp(llvm::StringRef name,
                                                   mlir::Type stdType) {
  auto moduleOp = cgf.getCIRGenModule().getModule();
  if (moduleOp.lookupSymbol<mlir::omp::PrivateClauseOp>(name))
    return;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  mlir::omp::PrivateClauseOp::create(builder, loc, name, stdType,
                                     mlir::omp::DataSharingClauseType::Private);
}

void OMPDataSharingProcessor::processStep1(
    llvm::ArrayRef<const OMPClause *> clauses,
    OMPPrivateClauseOps &clauseOps, mlir::Operation *insertBeforeOp) {
  // Collect private variables from all OMPPrivateClause nodes.
  for (const OMPClause *c : clauses) {
    const auto *privClause = dyn_cast<OMPPrivateClause>(c);
    if (!privClause)
      continue;
    for (const Expr *varExpr : privClause->varlist()) {
      const auto *dre =
          cast<DeclRefExpr>(varExpr->IgnoreParenImpCasts());
      const auto *vd = cast<VarDecl>(dre->getDecl());

      Address addr = cgf.getAddrOfLocalVar(vd);
      mlir::Value originalAddr = addr.getPointer();
      mlir::Type elementType = addr.getElementType();

      mlir::Type stdType = convertCIRTypeToStdType(elementType);
      if (!stdType)
        continue; // errorNYI already emitted

      std::string privatizerName = vd->getNameAsString() + ".privatizer";
      getOrCreatePrivateOp(privatizerName, stdType);

      entries.push_back({vd, originalAddr, elementType, privatizerName, {}});
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
