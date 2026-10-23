//===- LoopTermFold.cpp - Eliminate last use of IV in exit branch----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopTermFold.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "loop-term-fold"

STATISTIC(NumTermFold,
          "Number of terminating condition fold recognized and performed");

STATISTIC(NumMaxBTCInferred,
          "Number of loops with a backedge-taken bound inferred from inbounds "
          "accesses");

//===----------------------------------------------------------------------===//
// Backedge-taken-count bound from inbounds memory accesses.
//
// LoopTermFold needs to know that the IV it uses for the new exit test does
// not take the same value twice before the loop exits.  SCEV proves this with
// the no-self-wrap flag, but LSR builds its new IVs with plain
// 'getelementptr i8' and the flag is missing on the pointer IV.
// However, the original program still contains the information
// *before* LSR runs:
//
//   If a load or store executes on every iteration and its address is
//   computed by inbounds GEPs from a loop-invariant base, every address it
//   uses lies inside one allocated object (otherwise the address is poison
//   and the access is UB).  An allocated object is at most
//   SignedMax(IndexTy) bytes and does not cross the address space boundary,
//   so an access whose address advances by S bytes per iteration bounds the
//   backedge-taken count by SignedMax / |S|.
//
// LoopTermFoldPrepare computes this bound before LSR and records it in the
// loop metadata.  LoopTermFold consumes it, and accepts a candidate
// IV with constant step S' when the bound is below the IV's period
// 2^(BitWidth - ctz(S')).
//===----------------------------------------------------------------------===//

static cl::opt<bool> InferBoundFromInBounds(
    "loop-term-fold-infer-inbounds-bound", cl::Hidden, cl::init(true),
    cl::desc("Infer a backedge-taken count bound from inbounds accesses "
             "before LSR, to prove that LoopTermFold candidate IVs do not "
             "wrap"));

static cl::opt<unsigned> ExpansionBudgetOverride(
    "loop-term-fold-expansion-budget", cl::Hidden, cl::init(0),
    cl::desc("Preheader expansion budget for loop terminating condition "
             "folding (0 = target default)"));

static cl::opt<cl::boolOrDefault> RewriteExitUsesOpt(
    "loop-term-fold-rewrite-exit-uses", cl::Hidden,
    cl::desc("Fold the terminating condition even if the IV is used after "
             "the loop, by rewriting those uses to the IV's exit value "
             "(default: target hook)"));

static constexpr const char *MaxBTCAttr = "llvm.loop.term_fold.max_btc";

/// Return true if \p Ptr is computed from a loop-invariant pointer through
/// inbounds GEPs only, possibly through one header PHI whose backedge value
/// is itself an inbounds GEP chain back to the PHI.  All values of \p Ptr on
/// well-defined executions are then in bounds of the object the invariant
/// pointer is based on.
static bool isInBoundsOfInvariantObject(Value *Ptr, const Loop *L) {
  BasicBlock *Header = L->getHeader();
  bool SeenPHI = false;
  while (!L->isLoopInvariant(Ptr)) {
    if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
      if (!GEP->isInBounds())
        return false;
      Ptr = GEP->getPointerOperand();
      continue;
    }
    auto *PN = dyn_cast<PHINode>(Ptr);
    if (!PN || PN->getParent() != Header || SeenPHI)
      return false;
    Value *Next = PN->getIncomingValueForBlock(L->getLoopLatch());
    while (Next != PN) {
      auto *GEP = dyn_cast<GetElementPtrInst>(Next);
      if (!GEP || !GEP->isInBounds() || !L->contains(GEP))
        return false;
      Next = GEP->getPointerOperand();
    }
    SeenPHI = true;
    Ptr = PN->getIncomingValueForBlock(L->getLoopPreheader());
  }
  return true;
}

/// Compute an upper bound of the backedge-taken count of \p L that holds on
/// every execution without undefined behavior, using the inbounds accesses
/// that run on every iteration.
static std::optional<APInt>
inferMaxBTCFromInBoundsAccesses(Loop *L, LoopInfo &LI, DominatorTree &DT,
                                ScalarEvolution &SE) {
  if (!L->isInnermost() || !L->isLoopSimplifyForm())
    return std::nullopt;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  // The latch must be the only way out, so that once an iteration starts it
  // reaches the latch...
  if (L->getExitingBlock() != Latch)
    return std::nullopt;

  // ...with no cycle inside the iteration (an irreducible cycle is not a
  // subloop) and no instruction that may stop execution from reaching it
  // (non-returning or throwing calls, etc.).  Then every iteration that
  // starts runs every block that dominates the latch, and the exit test of
  // iteration BTC is reached only after all accesses of iterations 0..BTC.
  LoopBlocksDFS DFS(L);
  DFS.perform(&LI);
  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
    for (BasicBlock *Succ : successors(BB))
      if (Succ != Header && L->contains(Succ) &&
          DFS.getRPO(Succ) <= DFS.getRPO(BB))
        return std::nullopt;
    for (Instruction &I : *BB)
      if (!isGuaranteedToTransferExecutionToSuccessor(&I))
        return std::nullopt;
  }

  const DataLayout &DL = Header->getDataLayout();
  std::optional<APInt> Best;
  for (BasicBlock *BB : L->blocks()) {
    if (!DT.dominates(BB, Latch))
      continue;
    for (Instruction &I : *BB) {
      Value *Ptr = getLoadStorePointerOperand(&I);
      if (!Ptr)
        continue;
      if (auto *LdI = dyn_cast<LoadInst>(&I); LdI && LdI->isVolatile())
        continue;
      if (auto *StI = dyn_cast<StoreInst>(&I); StI && StI->isVolatile())
        continue;
      Type *PtrTy = Ptr->getType();
      unsigned N = DL.getIndexTypeSizeInBits(PtrTy);
      if (N != DL.getPointerTypeSizeInBits(PtrTy))
        continue;
      if (!isInBoundsOfInvariantObject(Ptr, L))
        continue;
      auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Ptr));
      if (!AR || AR->getLoop() != L || !AR->isAffine())
        continue;
      auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
      if (!Step || Step->getValue()->isZero() ||
          Step->getAPInt().getBitWidth() != N)
        continue;
      // All BTC+1 addresses lie in [Obj, Obj + Size] with
      // Size <= SignedMax(N), without crossing the address space boundary.
      // Consecutive addresses differ by S modulo 2^N; since |S| <= 2^(N-1)
      // and Size < 2^(N-1), the only possible linear difference is S itself,
      // so the addresses advance monotonically and BTC * |S| <= Size.
      APInt S = Step->getAPInt().abs(); // SignedMin stays 2^(N-1) unsigned.
      APInt Bound = APInt::getSignedMaxValue(N).udiv(S);
      if (!Best || Bound.ult(*Best))
        Best = Bound;
    }
  }
  return Best;
}

/// Rebuild the loop ID of \p L without the MaxBTCAttr entry, appending
/// \p Extra if non-null.  Returns nullptr if nothing but the self reference
/// would be left.
static MDNode *rebuildLoopID(Loop *L, MDNode *Extra) {
  LLVMContext &Ctx = L->getHeader()->getContext();
  SmallVector<Metadata *, 4> Ops;
  Ops.push_back(nullptr); // Self reference, filled in below.
  if (MDNode *Old = L->getLoopID())
    for (const MDOperand &Op : drop_begin(Old->operands())) {
      if (auto *N = dyn_cast<MDNode>(Op))
        if (N->getNumOperands() > 0)
          if (auto *S = dyn_cast<MDString>(N->getOperand(0)))
            if (S->getString() == MaxBTCAttr)
              continue;
      Ops.push_back(Op);
    }
  if (Extra)
    Ops.push_back(Extra);
  if (Ops.size() == 1)
    return nullptr;
  MDNode *New = MDNode::getDistinct(Ctx, Ops);
  New->replaceOperandWith(0, New);
  return New;
}

static bool runTermFoldPrepare(Loop *L, LoopInfo &LI, DominatorTree &DT,
                               ScalarEvolution &SE) {
  if (!InferBoundFromInBounds)
    return false;
  std::optional<APInt> MaxBTC = inferMaxBTCFromInBoundsAccesses(L, LI, DT, SE);
  if (!MaxBTC)
    return false;
  LLVM_DEBUG(dbgs() << "Inferred max backedge-taken count " << *MaxBTC
                    << " from inbounds accesses in loop " << L->getName()
                    << "\n");
  LLVMContext &Ctx = L->getHeader()->getContext();
  Metadata *Ops[] = {
      MDString::get(Ctx, MaxBTCAttr),
      ConstantAsMetadata::get(ConstantInt::get(Ctx, *MaxBTC))};
  L->setLoopID(rebuildLoopID(L, MDNode::get(Ctx, Ops)));
  ++NumMaxBTCInferred;
  return true;
}

/// Read the bound recorded by LoopTermFoldPrepare and remove it from the
/// loop metadata.  Sets \p Changed if metadata was removed.
static std::optional<APInt> takeMaxBTCHint(Loop *L, bool &Changed) {
  MDNode *N = findOptionMDForLoop(L, MaxBTCAttr);
  if (!N)
    return std::nullopt;
  std::optional<APInt> Result;
  if (N->getNumOperands() == 2)
    if (auto *C = mdconst::dyn_extract<ConstantInt>(N->getOperand(1)))
      Result = C->getValue();
  L->setLoopID(rebuildLoopID(L, nullptr));
  Changed = true;
  return Result;
}

/// Like isAlmostDeadIV(), but also allows users outside \p L. Those are
/// rewritten to the IV's exit value when the fold is performed.
static bool isDeadIVExceptExitUses(Loop *L, PHINode *PN, BasicBlock *Latch,
                                   Value *Cond) {
  Value *IncV = PN->getIncomingValueForBlock(Latch);
  for (User *U : PN->users())
    if (U != Cond && U != IncV && L->contains(cast<Instruction>(U)))
      return false;
  for (User *U : IncV->users())
    if (U != Cond && U != PN && L->contains(cast<Instruction>(U)))
      return false;
  return true;
}

/// Replace every use of \p V outside \p L with \p ExitS, expanded at
/// \p InsertPt (the preheader terminator).  The loop has a single exit at the
/// latch, so a use outside the loop only sees the value of the last
/// iteration, which is what \p ExitS computes.
static void replaceUsesOutsideLoop(Loop *L, Value *V, const SCEV *ExitS,
                                   SCEVExpander &Expander,
                                   Instruction *InsertPt) {
  Value *ExitV = nullptr;
  for (Use &U : make_early_inc_range(V->uses())) {
    auto *UserI = cast<Instruction>(U.getUser());
    if (L->contains(UserI))
      continue;
    if (!ExitV)
      ExitV = Expander.expandCodeFor(ExitS, V->getType(), InsertPt);
    U.set(ExitV);
  }
}

/// True if an IV with constant non-zero step \p Step cannot repeat a value
/// within MaxBTC + 1 iterations: its period is 2^(BitWidth - ctz(Step)).
static bool isBelowIVPeriod(const APInt &MaxBTC, const APInt &Step) {
  unsigned TZ = Step.countr_zero();
  if (TZ >= Step.getBitWidth())
    return false;
  return MaxBTC.getActiveBits() <= Step.getBitWidth() - TZ;
}

static std::optional<
    std::tuple<PHINode *, PHINode *, const SCEV *, bool, bool>>
canFoldTermCondOfLoop(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                      const LoopInfo &LI, const TargetTransformInfo &TTI,
                      const std::optional<APInt> &MaxBTCHint) {
  if (!L->isInnermost()) {
    LLVM_DEBUG(dbgs() << "Cannot fold on non-innermost loop\n");
    return std::nullopt;
  }
  // Only inspect on simple loop structure
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "Cannot fold on non-simple loop\n");
    return std::nullopt;
  }

  if (!SE.hasLoopInvariantBackedgeTakenCount(L)) {
    LLVM_DEBUG(dbgs() << "Cannot fold on backedge that is loop variant\n");
    return std::nullopt;
  }

  BasicBlock *LoopLatch = L->getLoopLatch();
  CondBrInst *BI = dyn_cast<CondBrInst>(LoopLatch->getTerminator());
  if (!BI)
    return std::nullopt;
  auto *TermCond = dyn_cast<ICmpInst>(BI->getCondition());
  if (!TermCond) {
    LLVM_DEBUG(
        dbgs() << "Cannot fold on branching condition that is not an ICmpInst");
    return std::nullopt;
  }
  if (!TermCond->hasOneUse()) {
    LLVM_DEBUG(
        dbgs()
        << "Cannot replace terminating condition with more than one use\n");
    return std::nullopt;
  }

  BinaryOperator *LHS = dyn_cast<BinaryOperator>(TermCond->getOperand(0));
  Value *RHS = TermCond->getOperand(1);
  if (!LHS || !L->isLoopInvariant(RHS))
    // We could pattern match the inverse form of the icmp, but that is
    // non-canonical, and this pass is running *very* late in the pipeline.
    return std::nullopt;

  // Find the IV used by the current exit condition.
  PHINode *ToFold;
  Value *ToFoldStart, *ToFoldStep;
  if (!matchSimpleRecurrence(LHS, ToFold, ToFoldStart, ToFoldStep))
    return std::nullopt;

  // Ensure the simple recurrence is a part of the current loop.
  if (ToFold->getParent() != L->getHeader())
    return std::nullopt;

  // If that IV isn't dead after we rewrite the exit condition in terms of
  // another IV, there's no point in doing the transform.
  // If the target allows it, uses after the loop are acceptable: they are
  // rewritten to the IV's exit value, whose cost is checked below.
  bool RewriteExitUses = RewriteExitUsesOpt == cl::BOU_UNSET
                             ? TTI.shouldLoopTermFoldRewriteExitUses()
                             : RewriteExitUsesOpt == cl::BOU_TRUE;
  bool HasExitUses = false;
  if (!isAlmostDeadIV(ToFold, LoopLatch, TermCond)) {
    if (!RewriteExitUses || L->getExitingBlock() != LoopLatch ||
        !isDeadIVExceptExitUses(L, ToFold, LoopLatch, TermCond)) {
      LLVM_DEBUG(dbgs() << "IV to fold has other uses\n");
      return std::nullopt;
    }
    HasExitUses = true;
  }

  // Inserting instructions in the preheader has a runtime cost, scale
  // the allowed cost with the loops trip count as best we can.
  const unsigned ExpansionBudget = [&]() {
    unsigned Budget = 2 * SCEVCheapExpansionBudget;
    if (ExpansionBudgetOverride)
      Budget = ExpansionBudgetOverride;
    else if (!L->getHeader()->getParent()->hasMinSize())
      if (unsigned TargetBudget = TTI.getLoopTermFoldExpansionBudget())
        Budget = TargetBudget;
    if (unsigned SmallTC = SE.getSmallConstantMaxTripCount(L))
      return std::min(Budget, SmallTC);
    if (std::optional<unsigned> SmallTC = getLoopEstimatedTripCount(L))
      return std::min(Budget, *SmallTC);
    // Unknown trip count, assume long running by default.
    return Budget;
  }();

  const SCEV *BECount = SE.getBackedgeTakenCount(L);
  SCEVExpander Expander(SE, "lsr_fold_term_cond");

  PHINode *ToHelpFold = nullptr;
  const SCEV *TermValueS = nullptr;
  bool MustDropPoison = false;
  auto InsertPt = L->getLoopPreheader()->getTerminator();
  for (PHINode &PN : L->getHeader()->phis()) {
    if (ToFold == &PN)
      continue;

    if (!SE.isSCEVable(PN.getType())) {
      LLVM_DEBUG(dbgs() << "IV of phi '" << PN
                        << "' is not SCEV-able, not qualified for the "
                           "terminating condition folding.\n");
      continue;
    }
    const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&PN));
    // Only speculate on affine AddRec
    if (!AddRec || !AddRec->isAffine()) {
      LLVM_DEBUG(dbgs() << "SCEV of phi '" << PN
                        << "' is not an affine add recursion, not qualified "
                           "for the terminating condition folding.\n");
      continue;
    }

    // Check that we can compute the value of AddRec on the exiting iteration
    // without soundness problems.  evaluateAtIteration internally needs
    // to multiply the stride of the iteration number - which may wrap around.
    // The issue here is subtle because computing the result accounting for
    // wrap is insufficient. In order to use the result in an exit test, we
    // must also know that AddRec doesn't take the same value on any previous
    // iteration. The simplest case to consider is a candidate IV which is
    // narrower than the trip count (and thus original IV), but this can
    // also happen due to non-unit strides on the candidate IVs.
    //
    // Without the no-self-wrap flag, a bound on the backedge-taken count
    // inferred before LSR (see LoopTermFoldPrepare) can prove the same
    // property for a constant step.
    bool NoSelfWrap = AddRec->hasNoSelfWrap();
    if (!NoSelfWrap && MaxBTCHint)
      if (auto *StepC = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE)))
        NoSelfWrap = isBelowIVPeriod(*MaxBTCHint, StepC->getAPInt());
    if (!NoSelfWrap || !SE.isKnownNonZero(AddRec->getStepRecurrence(SE))) {
      LLVM_DEBUG(dbgs() << "SCEV of phi '" << PN
                        << "' may wrap or has a zero step\n");
      continue;
    }

    const SCEVAddRecExpr *PostInc = AddRec->getPostIncExpr(SE);
    const SCEV *TermValueSLocal = PostInc->evaluateAtIteration(BECount, SE);
    if (!Expander.isSafeToExpand(TermValueSLocal)) {
      LLVM_DEBUG(
          dbgs() << "Is not safe to expand terminating value for phi node" << PN
                 << "\n");
      continue;
    }

    if (Expander.isHighCostExpansion(TermValueSLocal, L, ExpansionBudget, &TTI,
                                     InsertPt)) {
      LLVM_DEBUG(
          dbgs() << "Is too expensive to expand terminating value for phi node"
                 << PN << "\n");
      continue;
    }

    // The candidate IV may have been otherwise dead and poison from the
    // very first iteration.  If we can't disprove that, we can't use the IV.
    if (!mustExecuteUBIfPoisonOnPathTo(&PN, LoopLatch->getTerminator(), &DT)) {
      LLVM_DEBUG(dbgs() << "Can not prove poison safety for IV " << PN << "\n");
      continue;
    }

    // The candidate IV may become poison on the last iteration.  If this
    // value is not branched on, this is a well defined program.  We're
    // about to add a new use to this IV, and we have to ensure we don't
    // insert UB which didn't previously exist.
    bool MustDropPoisonLocal = false;
    Instruction *PostIncV =
        cast<Instruction>(PN.getIncomingValueForBlock(LoopLatch));
    if (!mustExecuteUBIfPoisonOnPathTo(PostIncV, LoopLatch->getTerminator(),
                                       &DT)) {
      LLVM_DEBUG(dbgs() << "Can not prove poison safety to insert use" << PN
                        << "\n");

      // If this is a complex recurrance with multiple instructions computing
      // the backedge value, we might need to strip poison flags from all of
      // them.
      if (PostIncV->getOperand(0) != &PN)
        continue;

      // In order to perform the transform, we need to drop the poison
      // generating flags on this instruction (if any).
      MustDropPoisonLocal = PostIncV->hasPoisonGeneratingFlags();
    }

    // We pick the last legal alternate IV.  We could expore choosing an optimal
    // alternate IV if we had a decent heuristic to do so.
    ToHelpFold = &PN;
    TermValueS = TermValueSLocal;
    MustDropPoison = MustDropPoisonLocal;
  }

  LLVM_DEBUG(if (ToFold && !ToHelpFold) dbgs()
                 << "Cannot find other AddRec IV to help folding\n";);

  LLVM_DEBUG(if (ToFold && ToHelpFold) dbgs()
             << "\nFound loop that can fold terminating condition\n"
             << "  BECount (SCEV): " << *SE.getBackedgeTakenCount(L) << "\n"
             << "  TermCond: " << *TermCond << "\n"
             << "  BrandInst: " << *BI << "\n"
             << "  ToFold: " << *ToFold << "\n"
             << "  ToHelpFold: " << *ToHelpFold << "\n");

  if (!ToFold || !ToHelpFold)
    return std::nullopt;

  // The exit values of the IV being removed must be safe and cheap to compute
  // in the preheader too.
  if (HasExitUses) {
    const auto *ToFoldAR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(ToFold));
    if (!ToFoldAR || ToFoldAR->getLoop() != L || !ToFoldAR->isAffine()) {
      LLVM_DEBUG(dbgs() << "IV to fold is not an affine AddRec of the loop\n");
      return std::nullopt;
    }
    for (const SCEV *ExitS :
         {ToFoldAR->evaluateAtIteration(BECount, SE),
          ToFoldAR->getPostIncExpr(SE)->evaluateAtIteration(BECount, SE)}) {
      if (!Expander.isSafeToExpand(ExitS) ||
          Expander.isHighCostExpansion(ExitS, L, ExpansionBudget, &TTI,
                                       InsertPt)) {
        LLVM_DEBUG(dbgs() << "Exit value of the IV to fold is too expensive "
                             "or unsafe to expand\n");
        return std::nullopt;
      }
    }
  }

  return std::make_tuple(ToFold, ToHelpFold, TermValueS, MustDropPoison,
                         HasExitUses);
}

static bool RunTermFold(Loop *L, ScalarEvolution &SE, DominatorTree &DT,
                        LoopInfo &LI, const TargetTransformInfo &TTI,
                        TargetLibraryInfo &TLI, MemorySSA *MSSA,
                        bool &MetadataOnlyChange) {
  MetadataOnlyChange = false;
  std::unique_ptr<MemorySSAUpdater> MSSAU;
  if (MSSA)
    MSSAU = std::make_unique<MemorySSAUpdater>(MSSA);

  // Always consume the hint so that it does not outlive this pass.
  bool HintRemoved = false;
  std::optional<APInt> MaxBTCHint = takeMaxBTCHint(L, HintRemoved);

  auto Opt = canFoldTermCondOfLoop(L, SE, DT, LI, TTI, MaxBTCHint);
  if (!Opt) {
    MetadataOnlyChange = HintRemoved;
    return false;
  }

  auto [ToFold, ToHelpFold, TermValueS, MustDrop, HasExitUses] = *Opt;

  NumTermFold++;

  BasicBlock *LoopPreheader = L->getLoopPreheader();
  BasicBlock *LoopLatch = L->getLoopLatch();

  (void)ToFold;
  LLVM_DEBUG(dbgs() << "To fold phi-node:\n"
                    << *ToFold << "\n"
                    << "New term-cond phi-node:\n"
                    << *ToHelpFold << "\n");

  Value *StartValue = ToHelpFold->getIncomingValueForBlock(LoopPreheader);
  (void)StartValue;
  Value *LoopValue = ToHelpFold->getIncomingValueForBlock(LoopLatch);

  // See comment in canFoldTermCondOfLoop on why this is sufficient.
  if (MustDrop)
    cast<Instruction>(LoopValue)->dropPoisonGeneratingFlags();

  // SCEVExpander for both use in preheader and latch
  SCEVExpander Expander(SE, "lsr_fold_term_cond");

  assert(Expander.isSafeToExpand(TermValueS) &&
         "Terminating value was checked safe in canFoldTerminatingCondition");

  // Create new terminating value at loop preheader
  Value *TermValue = Expander.expandCodeFor(TermValueS, ToHelpFold->getType(),
                                            LoopPreheader->getTerminator());

  LLVM_DEBUG(dbgs() << "Start value of new term-cond phi-node:\n"
                    << *StartValue << "\n"
                    << "Terminating value of new term-cond phi-node:\n"
                    << *TermValue << "\n");

  // Uses of the removed IV after the loop get its exit value instead, so the
  // IV becomes dead and DeleteDeadPHIs below removes it from the loop.
  if (HasExitUses) {
    const auto *ToFoldAR = cast<SCEVAddRecExpr>(SE.getSCEV(ToFold));
    const SCEV *BECount = SE.getBackedgeTakenCount(L);
    Value *IncV = ToFold->getIncomingValueForBlock(LoopLatch);
    Instruction *InsertPt = LoopPreheader->getTerminator();
    replaceUsesOutsideLoop(
        L, IncV,
        ToFoldAR->getPostIncExpr(SE)->evaluateAtIteration(BECount, SE),
        Expander, InsertPt);
    replaceUsesOutsideLoop(L, ToFold, ToFoldAR->evaluateAtIteration(BECount, SE),
                           Expander, InsertPt);
  }

  // Create new terminating condition at loop latch
  CondBrInst *BI = cast<CondBrInst>(LoopLatch->getTerminator());
  ICmpInst *OldTermCond = cast<ICmpInst>(BI->getCondition());
  IRBuilder<> LatchBuilder(LoopLatch->getTerminator());
  Value *NewTermCond =
      LatchBuilder.CreateICmp(CmpInst::ICMP_EQ, LoopValue, TermValue,
                              "lsr_fold_term_cond.replaced_term_cond");
  // Swap successors to exit loop body if IV equals to new TermValue
  if (BI->getSuccessor(0) == L->getHeader())
    BI->swapSuccessors();

  LLVM_DEBUG(dbgs() << "Old term-cond:\n"
                    << *OldTermCond << "\n"
                    << "New term-cond:\n"
                    << *NewTermCond << "\n");

  BI->setCondition(NewTermCond);

  Expander.clear();
  OldTermCond->eraseFromParent();
  DeleteDeadPHIs(L->getHeader(), &TLI, MSSAU.get());
  return true;
}

namespace {

class LoopTermFold : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid

  LoopTermFold();

private:
  bool runOnLoop(Loop *L, LPPassManager &LPM) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // end anonymous namespace

LoopTermFold::LoopTermFold() : LoopPass(ID) {
  initializeLoopTermFoldPass(*PassRegistry::getPassRegistry());
}

void LoopTermFold::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LoopSimplifyID);
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addPreserved<MemorySSAWrapperPass>();
}

bool LoopTermFold::runOnLoop(Loop *L, LPPassManager & /*LPM*/) {
  if (skipLoop(L))
    return false;

  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  const auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
      *L->getHeader()->getParent());
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(
      *L->getHeader()->getParent());
  auto *MSSAAnalysis = getAnalysisIfAvailable<MemorySSAWrapperPass>();
  MemorySSA *MSSA = nullptr;
  if (MSSAAnalysis)
    MSSA = &MSSAAnalysis->getMSSA();
  bool MetadataOnlyChange;
  bool Folded = RunTermFold(L, SE, DT, LI, TTI, TLI, MSSA, MetadataOnlyChange);
  return Folded || MetadataOnlyChange;
}

PreservedAnalyses LoopTermFoldPass::run(Loop &L, LoopAnalysisManager &AM,
                                        LoopStandardAnalysisResults &AR,
                                        LPMUpdater &) {
  bool MetadataOnlyChange;
  // Removing the hint metadata does not invalidate any analysis.
  if (!RunTermFold(&L, AR.SE, AR.DT, AR.LI, AR.TTI, AR.TLI, AR.MSSA,
                   MetadataOnlyChange))
    return PreservedAnalyses::all();

  auto PA = getLoopPassPreservedAnalyses();
  if (AR.MSSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

char LoopTermFold::ID = 0;

INITIALIZE_PASS_BEGIN(LoopTermFold, "loop-term-fold", "Loop Terminator Folding",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopTermFold, "loop-term-fold", "Loop Terminator Folding",
                    false, false)

Pass *llvm::createLoopTermFoldPass() { return new LoopTermFold(); }

//===----------------------------------------------------------------------===//
// LoopTermFoldPrepare
//===----------------------------------------------------------------------===//

namespace {

class LoopTermFoldPrepare : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid

  LoopTermFoldPrepare() : LoopPass(ID) {
    initializeLoopTermFoldPreparePass(*PassRegistry::getPassRegistry());
  }

private:
  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    return runTermFoldPrepare(L, LI, DT, SE);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    // Only loop metadata is changed.
    AU.setPreservesAll();
  }
};

} // end anonymous namespace

PreservedAnalyses LoopTermFoldPreparePass::run(Loop &L, LoopAnalysisManager &,
                                               LoopStandardAnalysisResults &AR,
                                               LPMUpdater &) {
  // Only loop metadata is changed, which no analysis depends on.
  runTermFoldPrepare(&L, AR.LI, AR.DT, AR.SE);
  return PreservedAnalyses::all();
}

char LoopTermFoldPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(LoopTermFoldPrepare, "loop-term-fold-prepare",
                      "Loop Terminator Folding Preparation", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopTermFoldPrepare, "loop-term-fold-prepare",
                    "Loop Terminator Folding Preparation", false, false)

Pass *llvm::createLoopTermFoldPreparePass() {
  return new LoopTermFoldPrepare();
}
