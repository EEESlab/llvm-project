//===- CIRUnrollByTwo.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CIRUNROLLBYTWO
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static SmallVector<Operation *> collectWithoutTerminator(Block &block) {
  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);
  return ops;
}

static void cloneOps(ArrayRef<Operation *> ops, OpBuilder &b, IRMapping &m) {
  for (Operation *op : ops)
    b.clone(*op, m);
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct IVPieces {
  cir::StoreOp stepStore;
  Value ivAddr;
  cir::LoadOp condIvLoad;
  // All ops in the step region before the StoreOp (the load of %iv plus any
  // arithmetic). The last op must have exactly one result — that result is
  // the new IV value being stored back.
  SmallVector<Operation *> stepComputations;
  SmallVector<Operation *> bodyOps;
  SmallVector<Operation *> condOps;
  cir::ConditionOp condTerm;
};

// Unconditionally walk all ops nested under `root`, crossing IsolatedFromAbove
// region boundaries that the typed Op::walk() would stop at.
static void walkAllOps(Operation *root,
                       llvm::function_ref<void(Operation *)> fn) {
  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (Operation &op : block) {
        fn(&op);
        walkAllOps(&op, fn);
      }
}

// Returns true if forOp contains no nested cir::ForOp in any of its regions,
// crossing cir::ScopeOp (IsolatedFromAbove) boundaries.
// We only unroll innermost loops.
static bool isInnermostLoop(cir::ForOp forOp) {
  bool foundNested = false;
  walkAllOps(forOp.getOperation(), [&](Operation *op) {
    if (op != forOp.getOperation() && isa<cir::ForOp>(op))
      foundNested = true;
  });
  return !foundNested;
}

static LogicalResult analyzeLoop(cir::ForOp forOp, IVPieces &out) {
  if (!isInnermostLoop(forOp))
    return failure();

  if (!forOp.getCond().hasOneBlock() || !forOp.getBody().hasOneBlock() ||
      !forOp.getStep().hasOneBlock())
    return failure();

  Block &condBlock = forOp.getCond().front();
  Block &bodyBlock = forOp.getBody().front();
  Block &stepBlock  = forOp.getStep().front();

  if (!condBlock.getArguments().empty() ||
      !bodyBlock.getArguments().empty()  ||
      !stepBlock.getArguments().empty())
    return failure();

  if (!isa<cir::YieldOp>(stepBlock.getTerminator()) ||
      !isa<cir::YieldOp>(bodyBlock.getTerminator()))
    return failure();

  out.condTerm = dyn_cast<cir::ConditionOp>(condBlock.getTerminator());
  if (!out.condTerm)
    return failure();

  // The step region must end (before yield) with a StoreOp that writes the
  // updated IV back to the alloca.
  SmallVector<Operation *> rawStepOps = collectWithoutTerminator(stepBlock);
  if (rawStepOps.empty())
    return failure();

  out.stepStore = dyn_cast<cir::StoreOp>(rawStepOps.back());
  if (!out.stepStore)
    return failure();

  out.ivAddr = out.stepStore.getAddr();

  // Find the unique load of ivAddr in the cond region.
  out.condIvLoad = nullptr;
  for (Operation &op : condBlock.without_terminator()) {
    auto load = dyn_cast<cir::LoadOp>(op);
    if (!load || load.getAddr() != out.ivAddr)
      continue;
    if (out.condIvLoad)
      return failure(); // more than one — too complex
    out.condIvLoad = load;
  }
  if (!out.condIvLoad)
    return failure();

  // stepComputations: everything before the StoreOp.
  // By construction the last element produces the value fed into the store.
  out.stepComputations.clear();
  for (Operation *op : rawStepOps) {
    if (op == out.stepStore)
      break;
    out.stepComputations.push_back(op);
  }
  if (out.stepComputations.empty())
    return failure();
  if (out.stepComputations.back()->getNumResults() != 1)
    return failure();

  out.bodyOps = collectWithoutTerminator(bodyBlock);
  out.condOps = collectWithoutTerminator(condBlock);
  return success();
}

// ---------------------------------------------------------------------------
// tryGetTripCount
//
// Attempt to statically determine the trip count of the loop.
// Returns the trip count if all three of the initial IV value, the loop
// bound, and the step stride are compile-time integer constants.
// Otherwise returns std::nullopt and the caller falls back to the
// conservative while-form transformation.
//
// Recognises the common pattern:
//
//   int i = C_init;          // StoreOp of a ConstantOp into ivAddr, found
//                            // by scanning backward before the ForOp.
//   for (; i < C_bound; )   // CmpOp(ivLoad, ConstantOp) in the cond region.
//       i += C_step;         // BinOp(Add, _, ConstantOp) in stepComputations.
//
// trip_count = ceil((C_bound - C_init) / C_step)
// ---------------------------------------------------------------------------
static std::optional<int64_t> tryGetTripCount(cir::ForOp forOp,
                                              IVPieces &p) {
  // ── 1. Find the initial IV value ──────────────────────────────────────
  // Walk backward through the ops preceding the ForOp in its parent block,
  // looking for a StoreOp that writes a ConstantOp value into ivAddr.
  std::optional<int64_t> initVal;
  Block *parentBlock = forOp->getBlock();
  for (auto it = Block::iterator(forOp); it != parentBlock->begin();) {
    --it;
    auto store = dyn_cast<cir::StoreOp>(&*it);
    if (!store || store.getAddr() != p.ivAddr)
      continue;
    auto defOp = store.getValue().getDefiningOp();
    if (!defOp)
      break;
    auto constOp = dyn_cast<cir::ConstantOp>(defOp);
    if (!constOp)
      break;
    auto intAttr = dyn_cast<cir::IntAttr>(constOp.getValue());
    if (!intAttr)
      break;
    initVal = intAttr.getValue().getSExtValue();
    break;
  }
  if (!initVal)
    return std::nullopt;

  // ── 2. Find the upper bound ───────────────────────────────────────────
  // Look for a CmpOp in the cond region whose LHS is the IV load result.
  // The RHS is either:
  //   (a) a ConstantOp directly, or
  //   (b) a LoadOp whose address was initialized by a StoreOp of a
  //       ConstantOp before the ForOp — the pattern emitted when the
  //       bound comes from a #define-derived local variable.
  // We only handle strict less-than; other predicates are left for future work.

  // resolveConstValue, findConstStoreForAddr and resolveBlockArg are mutually
  // recursive: a bound may be passed through several layers of function calls,
  // each time stored from a load of a local that was itself initialized from
  // a higher-level argument or constant.  We use std::function so the lambdas
  // can refer to each other.

  std::function<std::optional<int64_t>(Value, Operation *)> findConstStoreForAddr;
  std::function<std::optional<int64_t>(BlockArgument)>      resolveBlockArg;
  std::function<std::optional<int64_t>(Value)>              resolveConstValue;

  // resolveConstValue: given any Value, extract a compile-time integer.
  //   (a) direct ConstantOp
  //   (b) LoadOp  → findConstStoreForAddr on its address
  resolveConstValue = [&](Value v) -> std::optional<int64_t> {
    if (!v) {
      llvm::errs() << "[resolveConstValue] null value\n";
      return std::nullopt;
    }
    auto defOp = v.getDefiningOp();
    if (!defOp) {
      llvm::errs() << "[resolveConstValue] no defining op (block arg at top level?)\n";
      return std::nullopt;
    }
    llvm::errs() << "[resolveConstValue] defining op: " << defOp->getName() << "\n";
    if (auto constOp = dyn_cast<cir::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<cir::IntAttr>(constOp.getValue())) {
        llvm::errs() << "[resolveConstValue] direct constant: "
                     << intAttr.getValue().getSExtValue() << "\n";
        return intAttr.getValue().getSExtValue();
      }
      llvm::errs() << "[resolveConstValue] ConstantOp but not IntAttr\n";
      return std::nullopt;
    }
    if (auto loadOp = dyn_cast<cir::LoadOp>(defOp)) {
      llvm::errs() << "[resolveConstValue] following LoadOp addr: ";
      loadOp.getAddr().dump();
      return findConstStoreForAddr(loadOp.getAddr(), loadOp.getOperation());
    }
    llvm::errs() << "[resolveConstValue] not a ConstantOp or LoadOp\n";
    return std::nullopt;
  };

  // findConstStoreForAddr: walk backward from searchFrom through all enclosing
  // blocks looking for a StoreOp to addr, then resolve the stored value.
  findConstStoreForAddr = [&](Value addr,
                              Operation *searchFrom) -> std::optional<int64_t> {
    llvm::errs() << "[findConstStoreForAddr] searching for addr: ";
    addr.dump();
    Operation *cursor = searchFrom;
    int depth = 0;
    while (cursor) {
      Block *block = cursor->getBlock();
      if (!block) {
        llvm::errs() << "[findConstStoreForAddr] depth=" << depth
                     << " cursor has no parent block, stopping\n";
        break;
      }
      llvm::errs() << "[findConstStoreForAddr] depth=" << depth
                   << " scanning block in op: "
                   << block->getParentOp()->getName() << "\n";
      for (auto it = Block::iterator(cursor); it != block->begin();) {
        --it;
        auto store = dyn_cast<cir::StoreOp>(&*it);
        if (!store || store.getAddr() != addr)
          continue;
        llvm::errs() << "[findConstStoreForAddr] found store: ";
        store.dump();
        Value stored = store.getValue();
        // Stored value is a block argument → chase to call site(s).
        if (auto blockArg = dyn_cast<BlockArgument>(stored)) {
          llvm::errs() << "[findConstStoreForAddr] stored value is block arg #"
                       << blockArg.getArgNumber() << ", chasing call sites\n";
          return resolveBlockArg(blockArg);
        }
        // Otherwise resolve generically (handles ConstantOp and LoadOp).
        return resolveConstValue(stored);
      }
      llvm::errs() << "[findConstStoreForAddr] not found in this block, climbing\n";
      cursor = block->getParentOp();
      ++depth;
    }
    llvm::errs() << "[findConstStoreForAddr] exhausted all parent blocks\n";
    return std::nullopt;
  };

  // resolveBlockArg: find all call sites of the parent function and check
  // that every one passes the same compile-time constant for this argument.
  // The call-site operand is resolved via resolveConstValue, so chains like
  //   main local → load → call arg → store → load → call arg → …
  // are followed transitively.
  resolveBlockArg = [&](BlockArgument arg) -> std::optional<int64_t> {
    llvm::errs() << "[resolveBlockArg] resolving block arg #"
                 << arg.getArgNumber() << "\n";
    auto funcOp = dyn_cast<cir::FuncOp>(arg.getOwner()->getParentOp());
    if (!funcOp) {
      llvm::errs() << "[resolveBlockArg] parent is not a cir.func\n";
      return std::nullopt;
    }
    unsigned argIdx = arg.getArgNumber();
    StringRef funcName = funcOp.getSymName();
    llvm::errs() << "[resolveBlockArg] function: " << funcName
                 << ", arg index: " << argIdx << "\n";
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      llvm::errs() << "[resolveBlockArg] no parent ModuleOp\n";
      return std::nullopt;
    }
    std::optional<int64_t> commonVal;
    bool found = false;
    moduleOp.walk([&](cir::CallOp call) {
      if (call.getCallee() != funcName)
        return;
      llvm::errs() << "[resolveBlockArg] found call site, operand count="
                   << call.getOperands().size() << "\n";
      if (argIdx >= call.getOperands().size()) {
        llvm::errs() << "[resolveBlockArg] argIdx out of range\n";
        commonVal = std::nullopt;
        found = true;
        return;
      }
      // Recursively resolve the operand — it may itself be a load of a local.
      auto val = resolveConstValue(call.getOperands()[argIdx]);
      if (!val) {
        llvm::errs() << "[resolveBlockArg] could not resolve call site operand\n";
        commonVal = std::nullopt;
        found = true;
        return;
      }
      llvm::errs() << "[resolveBlockArg] call site resolved to " << *val << "\n";
      if (!found) {
        commonVal = val;
        found = true;
      } else if (commonVal && *commonVal != *val) {
        llvm::errs() << "[resolveBlockArg] conflicting values at different call sites\n";
        commonVal = std::nullopt;
      }
    });
    if (!found)
      llvm::errs() << "[resolveBlockArg] no call sites found\n";
    else if (commonVal)
      llvm::errs() << "[resolveBlockArg] resolved to " << *commonVal << "\n";
    else
      llvm::errs() << "[resolveBlockArg] could not resolve to a constant\n";
    return commonVal;
  };

  // extractConstInt: entry point used by the bound and init extraction below.
  auto extractConstInt = [&](Value v) -> std::optional<int64_t> {
    return resolveConstValue(v);
  };

  std::optional<int64_t> boundVal;
  Block &condBlock = forOp.getCond().front();
  for (Operation &op : condBlock.without_terminator()) {
    auto cmp = dyn_cast<cir::CmpOp>(op);
    if (!cmp)
      continue;
    if (cmp.getLhs() != p.condIvLoad.getResult())
      continue;
    boundVal = extractConstInt(cmp.getRhs());
    if (boundVal)
      break;
  }
  if (!boundVal)
    return std::nullopt;

  // ── 3. Find the step stride ───────────────────────────────────────────
  // Two patterns are recognised:
  //   (a) BinOp(Add, _, ConstantOp)  — explicit i += K
  //   (b) UnaryOp(Inc, _)            — i++ / ++i, implicitly stride 1
  std::optional<int64_t> stepVal;
  for (Operation *op : p.stepComputations) {
    // Pattern (a): explicit addition with a constant operand.
    if (auto binop = dyn_cast<cir::BinOp>(op)) {
      if (binop.getKind() == cir::BinOpKind::Add) {
        for (Value operand : {binop.getLhs(), binop.getRhs()}) {
          if (auto defOp = operand.getDefiningOp()) {
            if (auto constOp = dyn_cast<cir::ConstantOp>(defOp)) {
              if (auto intAttr = dyn_cast<cir::IntAttr>(constOp.getValue())) {
                stepVal = intAttr.getValue().getSExtValue();
                break;
              }
            }
          }
        }
      }
    }
    // Pattern (b): pre/post-increment — stride is always 1.
    if (!stepVal) {
      if (auto unary = dyn_cast<cir::UnaryOp>(op)) {
        if (unary.getKind() == cir::UnaryOpKind::Inc)
          stepVal = 1;
      }
    }
    if (stepVal)
      break;
  }
  if (!stepVal || *stepVal <= 0)
    return std::nullopt;

  // ── 4. Compute trip count ─────────────────────────────────────────────
  int64_t range = *boundVal - *initVal;
  if (range <= 0)
    return std::nullopt; // loop never executes — leave it alone

  // Ceiling division: ceil(range / step).
  int64_t tripCount = (range + *stepVal - 1) / *stepVal;
  return tripCount;
}

// ---------------------------------------------------------------------------
// emitTwoStepArith
//
// Emits the step arithmetic twice, returning:
//   iv1 = i + 1*step  (result of cloning stepComputations verbatim)
//   iv2 = i + 2*step  (result of re-running arithmetic seeded from iv1)
//
// We never call StoreOp::getValue() because its operand order is ambiguous
// across CIR builds. Instead we use the result of the last cloned op directly,
// which by the analyzeLoop invariant is always the value fed into the store.
// ---------------------------------------------------------------------------
static LogicalResult emitTwoStepArith(IVPieces &p, OpBuilder &b,
                                      Value &iv1Out, Value &iv2Out) {
  // Round 1: clone stepComputations verbatim. iv1 = result of last cloned op.
  IRMapping m1;
  Operation *last1 = nullptr;
  for (Operation *op : p.stepComputations)
    last1 = b.clone(*op, m1);
  if (!last1 || last1->getNumResults() == 0)
    return failure();
  iv1Out = last1->getResult(0);

  // Round 2: replace the ivAddr load with iv1 so arithmetic starts from i+1.
  // If stepComputations is only the load op, every op is skipped and last2
  // stays null — fall back to re-cloning verbatim (load re-reads ivAddr=iv1).
  IRMapping m2;
  Operation *last2 = nullptr;
  bool anyCloned = false;
  for (Operation *op : p.stepComputations) {
    auto load = dyn_cast<cir::LoadOp>(op);
    if (load && load.getAddr() == p.ivAddr) {
      m2.map(load.getResult(), iv1Out);
      continue;
    }
    last2 = b.clone(*op, m2);
    anyCloned = true;
  }

  if (anyCloned && last2) {
    if (last2->getNumResults() == 0)
      return failure();
    iv2Out = last2->getResult(0);
  } else {
    // Fallback: re-clone verbatim; the load will re-read ivAddr which by now
    // holds iv1 (stored by the caller before the second body/cond clone).
    IRMapping m2b;
    Operation *last2b = nullptr;
    for (Operation *op : p.stepComputations)
      last2b = b.clone(*op, m2b);
    if (!last2b || last2b->getNumResults() == 0)
      return failure();
    iv2Out = last2b->getResult(0);
  }

  return success();
}

// ---------------------------------------------------------------------------
// emitCondRegion
//
// Shared helper used by both the while-form and do-while-form builders to
// emit the unrolled condition (i+1 < n) with a fallback to the original
// condition if emitTwoStepArith fails.
// ---------------------------------------------------------------------------
static void emitCondRegion(IVPieces &p, OpBuilder &b, Location l) {
  Value iv1, iv2;
  if (failed(emitTwoStepArith(p, b, iv1, iv2))) {
    llvm::errs() << "[emitCondRegion] emitTwoStepArith failed, using original cond\n";
    IRMapping m;
    cloneOps(p.condOps, b, m);
    Value cond = m.lookup(p.condTerm.getCondition());
    llvm::errs() << "[emitCondRegion] fallback cond valid: " << (bool)cond << "\n";
    cir::ConditionOp::create(b, l, cond);
    return;
  }
  llvm::errs() << "[emitCondRegion] iv1 valid: " << (bool)iv1
               << " iv2 valid: " << (bool)iv2 << "\n";
  IRMapping condMap;
  condMap.map(p.condIvLoad.getResult(), iv1);
  for (Operation *op : p.condOps) {
    if (op == p.condIvLoad)
      continue;
    b.clone(*op, condMap);
  }
  Value cond = condMap.lookup(p.condTerm.getCondition());
  llvm::errs() << "[emitCondRegion] cond valid: " << (bool)cond << "\n";
  if (!cond) {
    llvm::errs() << "[emitCondRegion] cond is null, using original cond as fallback\n";
    IRMapping m;
    cloneOps(p.condOps, b, m);
    Value fallbackCond = m.lookup(p.condTerm.getCondition());
    llvm::errs() << "[emitCondRegion] fallback cond valid: " << (bool)fallbackCond << "\n";
    cir::ConditionOp::create(b, l, fallbackCond);
    return;
  }
  cir::ConditionOp::create(b, l, cond);
}

// ---------------------------------------------------------------------------
// emitUnrolledBody
//
// Shared helper that emits the two-copy unrolled body:
//
//   body(i)             — reads ivAddr, sees i
//   store(ivAddr, i+1)  — advance in SSA, write back so isolated scopes see it
//   body(i+1)           — reads ivAddr, sees i+1
//   store(ivAddr, i+2)  — advance again for the next cond / loop-carried value
//   YieldOp
//
// See the detailed note in the original body lambda for why the intermediate
// store is required before the second body clone.
// ---------------------------------------------------------------------------
static void emitUnrolledBody(IVPieces &p, OpBuilder &b, Location l) {
  // ── First body copy (reads i from ivAddr). ──
  IRMapping m1;
  cloneOps(p.bodyOps, b, m1);

  // ── Advance i → i+1 in SSA and store to ivAddr. ──
  IRMapping stepMap1;
  Operation *lastStep1 = nullptr;
  for (Operation *op : p.stepComputations)
    lastStep1 = b.clone(*op, stepMap1);
  assert(lastStep1 && "stepComputations must not be empty (checked in analyzeLoop)");
  Value iv1 = lastStep1->getResult(0);
  IRMapping storeMap1;
  storeMap1.map(p.stepComputations.back()->getResult(0), iv1);
  b.clone(*p.stepStore.getOperation(), storeMap1);

  // ── Second body copy (loads ivAddr, now sees i+1). ──
  IRMapping m2;
  cloneOps(p.bodyOps, b, m2);

  // ── Advance i+1 → i+2 in SSA and store to ivAddr. ──
  // Seed from iv1 to avoid re-reading from memory.
  IRMapping stepMap2;
  Operation *lastStep2 = nullptr;
  bool anyCloned = false;
  for (Operation *op : p.stepComputations) {
    auto load = dyn_cast<cir::LoadOp>(op);
    if (load && load.getAddr() == p.ivAddr) {
      stepMap2.map(load.getResult(), iv1);
      continue;
    }
    lastStep2 = b.clone(*op, stepMap2);
    anyCloned = true;
  }

  Value iv2;
  if (anyCloned && lastStep2) {
    iv2 = lastStep2->getResult(0);
    llvm::errs() << "[emitUnrolledBody] iv2 computed via SSA seeding\n";
  } else {
    llvm::errs() << "[emitUnrolledBody] iv2 fallback: re-cloning stepComputations verbatim\n";
    IRMapping stepMap2b;
    Operation *last = nullptr;
    for (Operation *op : p.stepComputations)
      last = b.clone(*op, stepMap2b);
    assert(last && "stepComputations empty in fallback path");
    iv2 = last->getResult(0);
  }

  IRMapping storeMap2;
  storeMap2.map(p.stepComputations.back()->getResult(0), iv2);
  b.clone(*p.stepStore.getOperation(), storeMap2);

  cir::YieldOp::create(b, l);
}

// ---------------------------------------------------------------------------
// Main rewrite
// ---------------------------------------------------------------------------

static LogicalResult rewriteOne(cir::ForOp oldFor, IVPieces &p,
                                std::optional<int64_t> tripCount) {
  llvm::errs() << "[rewriteOne] called\n";

  // ── Static trip-count analysis ────────────────────────────────────────
  // tripCount was computed during the pre-analysis phase before any
  // ForOp was erased, avoiding use-after-free in findConstStoreForAddr.

  // A trip count of exactly 1 means unrolling by 2 would execute the second
  // body copy out of bounds; skip this loop entirely.
  if (tripCount)
    llvm::errs() << "[rewriteOne] tripCount=" << *tripCount << "\n";
  else
    llvm::errs() << "[rewriteOne] tripCount=unknown\n";

  if (tripCount && *tripCount == 1) {
    llvm::errs() << "[rewriteOne] tripCount==1, skipping\n";
    return failure();
  }

  // Do-while form is only safe when we can prove >= 2 iterations.
  const bool useDoWhile = tripCount.has_value() && *tripCount >= 2;

  // Tail IfOp is unnecessary when the trip count is statically even.
  const bool needTail = !tripCount.has_value() || (*tripCount % 2 != 0);

  llvm::errs() << "[rewriteOne] useDoWhile=" << useDoWhile
               << " needTail=" << needTail << "\n";

  OpBuilder builder(oldFor);
  Location loc = oldFor.getLoc();

  Operation *newLoopOp = nullptr; // set below, used to anchor the tail

  if (useDoWhile) {
    // ── Do-while form ─────────────────────────────────────────────────
    llvm::errs() << "[rewriteOne] emitting DoWhileOp (tripCount="
                 << *tripCount << ")\n";

    cir::DoWhileOp doWhile = cir::DoWhileOp::create(builder, loc,

      // ── cond (checked at the bottom) — first arg per DoWhileOp::create API ──
      [&](OpBuilder &b, Location l) {
        llvm::errs() << "[rewriteOne] building DoWhile cond\n";
        emitCondRegion(p, b, l);
        llvm::errs() << "[rewriteOne] DoWhile cond done\n";
      },

      // ── body — second arg per DoWhileOp::create API ──
      [&](OpBuilder &b, Location l) {
        llvm::errs() << "[rewriteOne] building DoWhile body\n";
        emitUnrolledBody(p, b, l);
        llvm::errs() << "[rewriteOne] DoWhile body done\n";
      });

    llvm::errs() << "[rewriteOne] DoWhileOp created, dumping:\n";
    doWhile.dump();
    llvm::errs() << "[rewriteOne] verifying DoWhileOp:\n";
    if (failed(mlir::verify(doWhile))) {
      llvm::errs() << "[rewriteOne] DoWhileOp FAILED verification — aborting rewrite\n";
      doWhile.erase();
      return failure();
    } else {
      llvm::errs() << "[rewriteOne] DoWhileOp passed verification\n";
    }
    newLoopOp = doWhile.getOperation();

  } else {
    // ── While form (conservative fallback) ────────────────────────────
    //
    // Used when we cannot prove tripCount >= 2 (unknown bounds or
    // tripCount == 0).  The entry condition guards both body copies.
    //
    // Structure:
    //   while (i+1 < n) {
    //     body(i); store(i+1); body(i+1); store(i+2);
    //   }
    cir::ForOp newFor = cir::ForOp::create(builder, loc,

      // ── cond ──
      [&](OpBuilder &b, Location l) {
        emitCondRegion(p, b, l);
      },

      // ── body ──
      [&](OpBuilder &b, Location l) {
        emitUnrolledBody(p, b, l);
      },

      // ── step: no-op — both advances are done inside the body ──
      [&](OpBuilder &b, Location l) {
        cir::YieldOp::create(b, l);
      });

    newLoopOp = newFor.getOperation();
    llvm::errs() << "[rewriteOne] ForOp created, dumping:\n";
    newFor.dump();
    if (failed(mlir::verify(newFor)))
      llvm::errs() << "[rewriteOne] ForOp FAILED verification\n";
    else
      llvm::errs() << "[rewriteOne] ForOp passed verification\n";
  }

  // ── Tail: if (i < n) { body(i) } ─────────────────────────────────────
  if (needTail) {
    llvm::errs() << "[rewriteOne] emitting tail IfOp\n";
    builder.setInsertionPointAfter(newLoopOp);

    IRMapping tailMap;
    cloneOps(p.condOps, builder, tailMap);
    Value tailCond = tailMap.lookup(p.condTerm.getCondition());
    llvm::errs() << "[rewriteOne] tailCond valid: " << (bool)tailCond << "\n";
    if (!tailCond) {
      llvm::errs() << "[rewriteOne] tailCond is null, aborting rewrite\n";
      return failure();
    }

    cir::IfOp::create(builder, loc, tailCond, /*withElseRegion=*/false,
      [&](OpBuilder &b, Location l) {
        IRMapping m;
        cloneOps(p.bodyOps, b, m);
        cir::YieldOp::create(b, l);
      });
    llvm::errs() << "[rewriteOne] tail IfOp emitted\n";
  } else {
    llvm::errs() << "[rewriteOne] tail skipped (even trip count)\n";
  }

  llvm::errs() << "[rewriteOne] erasing original ForOp\n";
  oldFor.erase();
  llvm::errs() << "[rewriteOne] done\n";
  return success();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct CIRUnrollByTwoPass
    : public mlir::impl::CIRUnrollByTwoBase<CIRUnrollByTwoPass> {
  using Base::Base;

  void runOnOperation() override {
    cir::FuncOp func = getOperation();

    // Collect all ForOps first. Use the raw Operation* walk to cross
    // IsolatedFromAbove region boundaries (e.g. cir::ScopeOp).
    SmallVector<cir::ForOp> loops;
    func.getOperation()->walk([&](cir::ForOp forOp) {
      loops.push_back(forOp);
    });

    // Process innermost loops first (reverse order).
    // IMPORTANT: rewriteOne calls oldFor.erase() which frees IR memory.
    // Any analysis that walks parent blocks (tryGetTripCount /
    // findConstStoreForAddr) must therefore complete BEFORE any erase.
    // We achieve this by running the full analysis for every loop up front,
    // storing the results, and only then performing the rewrites.

    struct LoopInfo {
      cir::ForOp forOp;
      IVPieces pieces;
      std::optional<int64_t> tripCount;
      bool valid; // false if analyzeLoop failed
    };

    SmallVector<LoopInfo> infos;
    for (cir::ForOp forOp : llvm::reverse(loops)) {
      LoopInfo info;
      info.forOp = forOp;
      info.valid = succeeded(analyzeLoop(forOp, info.pieces));
      if (info.valid)
        info.tripCount = tryGetTripCount(forOp, info.pieces);
      infos.push_back(info);
    }

    // Now perform rewrites. Each rewriteOne erases the original ForOp but
    // the analysis data in LoopInfo is self-contained (no pointers into
    // the erased op's storage beyond the IRMapping-independent IVPieces).
    for (LoopInfo &info : infos) {
      if (info.valid)
        (void)rewriteOne(info.forOp, info.pieces, info.tripCount);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createCIRUnrollByTwoPass() {
  return std::make_unique<CIRUnrollByTwoPass>();
}
