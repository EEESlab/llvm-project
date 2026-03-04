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

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H

#include "CIRGenBuilder.h"
#include "CIRGenValue.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class OMPClause;
class VarDecl;
} // namespace clang

namespace clang::CIRGen {

class CIRGenFunction;

/// Accumulated privatization operands to attach to an OpenMP op.
struct OMPPrivateClauseOps {
  llvm::SmallVector<mlir::Value> privateVars;
  llvm::SmallVector<mlir::Attribute> privateSyms;
};

/// Per-variable metadata collected during clause processing.
struct OMPPrivateVarEntry {
  const VarDecl *varDecl;
  mlir::Value originalAddr; // !cir.ptr<T> alloca
  mlir::Type elementType;   // CIR element type T
  std::string privatizerName;
  mlir::Value blockArg; // !llvm.ptr block arg, filled by addBlockArgs()
};

/// Centralizes OpenMP data sharing (private/firstprivate/lastprivate)
/// processing for CIR codegen.
///
/// Usage (inspired by Flang's DataSharingProcessor):
///
///   OMPDataSharingProcessor dsp(cgf, builder, loc);
///   dsp.processStep1(clauses, clauseOps, op);
///   // ... attach clauseOps to op ...
///   dsp.addBlockArgs(block);
///   {
///     auto guard = dsp.applyRemapping();
///     // ... emit body ...
///   }
class OMPDataSharingProcessor {
public:
  /// RAII guard that restores localDeclMap entries on destruction.
  class RemapGuard {
    CIRGenFunction &cgf;
    llvm::SmallVector<std::pair<const VarDecl *, Address>> savedAddrs;

  public:
    RemapGuard(CIRGenFunction &cgf,
               llvm::SmallVector<std::pair<const VarDecl *, Address>> saved);
    ~RemapGuard();
    RemapGuard(RemapGuard &&other) noexcept;
    RemapGuard &operator=(RemapGuard &&) = delete;
    RemapGuard(const RemapGuard &) = delete;
    RemapGuard &operator=(const RemapGuard &) = delete;
  };

  OMPDataSharingProcessor(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                          mlir::Location loc);

  /// Step 1: Collect private vars from clauses, create module-level
  /// omp.private ops, populate clauseOps with !llvm.ptr casts inserted
  /// before \p insertBeforeOp for SSA dominance.
  void processStep1(llvm::ArrayRef<const OMPClause *> clauses,
                    OMPPrivateClauseOps &clauseOps,
                    mlir::Operation *insertBeforeOp);

  /// Add !llvm.ptr block arguments to \p block for each private var.
  void addBlockArgs(mlir::Block &block);

  /// Insert !llvm.ptr → !cir.ptr casts at the current insertion point
  /// and remap localDeclMap entries. Returns an RAII guard that restores
  /// the original mappings on destruction.
  RemapGuard applyRemapping();

  bool hasPrivateVars() const { return !entries.empty(); }

  llvm::ArrayRef<OMPPrivateVarEntry> getEntries() const { return entries; }

private:
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;
  mlir::Location loc;
  llvm::SmallVector<OMPPrivateVarEntry> entries;

  /// Convert a CIR element type to the corresponding standard MLIR type
  /// for use in omp.private op.
  mlir::Type convertCIRTypeToStdType(mlir::Type cirType);

  /// Create or reuse an omp.private op at module level.
  void getOrCreatePrivateOp(llvm::StringRef name, mlir::Type stdType);
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
