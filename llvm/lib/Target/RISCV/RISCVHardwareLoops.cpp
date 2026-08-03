//===-- RISCVHardwareLoops.cpp - CORE-V hardware loops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower the generic hardware-loop pseudos produced by ISel into CORE-V
// (XCVhwlp) hardware-loop setup instructions, removing the corresponding
// software-managed loop control instructions. Conversion is best-effort: a loop
// failing any check below keeps its pseudos and their software expansion.
//
// A candidate loop is recognised by:
//   * PseudoCVHWLoopSetup / PseudoCVHWLoopSetupImm in the preheader,
//   * PseudoCVHWLoopEnd in the latch, defining a "continue" flag,
//   * a conditional branch in the latch consuming that flag, optionally
//     followed by an unconditional branch to the other successor.
//
// Two encodings are available:
//   * cv.setup / cv.setupi   - one instruction; lpstart is fixed at PC + 4.
//   * cv.starti + cv.endi +
//     cv.count / cv.counti   - three instructions, but a 12-bit loop end offset
//                              instead of the 5 bits of cv.setupi.
//
// Both are emitted as the last instructions of the preheader, which is required
// to be the layout predecessor of the header, and are preceded by
//
//     .p2align 2
//     .option push
//     .option norvc
//
// so that every PC-relative operand is known here and its range can be checked
// rather than left to the assembler. The matching .option pop is emitted at the
// top of the exit block, which is moved to directly follow the latch.
//
// lpend is a fixed address, so the body must have a statically known byte size.
// Branches are rejected outright rather than checked against the loop bounds,
// which restricts the body to a run of blocks connected only by fallthrough.
//
// CV32E40P provides two hardware-loop register sets. Innermost loops use set 0;
// the direct parent of an innermost loop may use set 1.
//
// This pass must run pre-emit: it depends on the final machine block layout, so
// it has to follow block placement and branch folding, and it assumes the
// post-RA pseudos have already been expanded.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "riscv-hardware-loops"

STATISTIC(NumHardwareLoops, "Number of CORE-V hardware loops generated");

static cl::opt<bool>
    DisableHardwareLoops("hwloops-disable", cl::Hidden, cl::init(false),
                         cl::desc("Disable CORE-V hardware loop generation"));

static cl::opt<bool> ForceLongSetup(
    "cvhwloop-force-long-setup", cl::Hidden, cl::init(false),
    cl::desc("Always use cv.starti/cv.endi/cv.count instead of cv.setup"));

static cl::opt<bool> PadShortBodies(
    "cvhwloop-pad-short-bodies", cl::Hidden, cl::init(false),
    cl::desc("Pad hardware loop bodies below the minimum length with nops"));

//===----------------------------------------------------------------------===//
// Encoding parameters
//===----------------------------------------------------------------------===//

namespace {

/// Width of every instruction inside the no-RVC region.
constexpr unsigned UncompressedInstrSize = 4;

/// Scale of the PC-relative loop start/end immediates, in bytes per encoded
/// unit. CV32E40P computes the target as PC + (uimm << 2).
constexpr unsigned CVEndOffsetScale = 4;

/// Widths of the PC-relative immediates.
constexpr unsigned CVSetupImmEndOffsetBits = 5;  ///< cv.setupi uimmL
constexpr unsigned CVSetupRegEndOffsetBits = 12; ///< cv.setup  uimmL
constexpr unsigned CVStartiOffsetBits = 12;      ///< cv.starti uimmL
constexpr unsigned CVEndiOffsetBits = 12;        ///< cv.endi   uimmL

/// Width of the trip-count immediate in cv.setupi (uimmS) and cv.counti.
constexpr unsigned CVCountBits = 12;

/// CV32E40P requires a hardware loop body of at least three instructions.
constexpr unsigned CVMinBodyInstructions = 3;

/// CV32E40P requires the outer loop end address to be at least eight bytes
/// beyond the inner loop end address.
constexpr unsigned CVMinNestedEndSeparation = 8;

/// Byte distance from each setup instruction to the header label. The setup
/// sequence is last in the preheader and the header follows it directly, so the
/// distance is just the width of what lies between — and inside the no-RVC
/// region that is four bytes per instruction:
///
///     | cv.setup[i] | Header ... Latch | Exit
///     | cv.starti | cv.endi | cv.count[i] | Header ... Latch | Exit
constexpr unsigned CVSetupToHeader = 1 * UncompressedInstrSize;
constexpr unsigned CVEndiToHeader = 2 * UncompressedInstrSize;
constexpr unsigned CVStartiToHeader = 3 * UncompressedInstrSize;

//===----------------------------------------------------------------------===//
// Hardware loop candidate description
//===----------------------------------------------------------------------===//

struct CVHWLoopCandidate {
  MachineLoop *Loop = nullptr;

  MachineBasicBlock *Preheader = nullptr;
  MachineBasicBlock *Header = nullptr;
  MachineBasicBlock *Latch = nullptr;
  MachineBasicBlock *Exit = nullptr;

  MachineInstr *Setup = nullptr;
  MachineInstr *LoopEnd = nullptr;
  MachineInstr *LoopBranch = nullptr;

  /// The latch's second terminator. Null means the latch falls through
  /// to one of its successors.
  MachineInstr *UncondBranch = nullptr;

  /// Snapshot of the loop's blocks. MachineLoopInfo is invalidated as soon as
  /// the first backedge is removed, so it cannot be queried during conversion.
  SmallPtrSet<MachineBasicBlock *, 8> Blocks;
};

struct CVHWLoopCount {
  bool IsImmediate = false;
  uint64_t Immediate = 0;
  Register Reg;
};

/// Measured properties of a loop body.
struct CVHWLoopBody {
  /// Number of real (non-zero-sized) instructions.
  unsigned NumInstructions = 0;
  /// Encoded size of the body in bytes.
  uint64_t SizeInBytes = 0;
  /// Set when an already-converted inner loop's end marker was seen.
  bool SawInnerEnd = false;
  /// Bytes between the last inner end marker and the end of the body.
  uint64_t BytesAfterInnerEnd = 0;
};

class RISCVHardwareLoops : public MachineFunctionPass {
public:
  static char ID;

  RISCVHardwareLoops() : MachineFunctionPass(ID) {
    initializeRISCVHardwareLoopsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "RISC-V CORE-V hardware loops";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const RISCVInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  std::optional<CVHWLoopCandidate> analyzeLoop(MachineLoop &ML) const;

  bool processLoopTree(MachineLoop &ML);

  bool convertCandidate(CVHWLoopCandidate &Candidate, unsigned LoopID,
                        bool IsOuter);
};

} // end anonymous namespace

char RISCVHardwareLoops::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVHardwareLoops, DEBUG_TYPE,
                      "RISC-V CORE-V hardware loops", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(RISCVHardwareLoops, DEBUG_TYPE,
                    "RISC-V CORE-V hardware loops", false, false)

FunctionPass *llvm::createRISCVHardwareLoopsPass() {
  return new RISCVHardwareLoops();
}

//===----------------------------------------------------------------------===//
// Generic helpers
//===----------------------------------------------------------------------===//

/// True if \p MI is branching to \p Target.
static bool branchTargets(const MachineInstr &MI,
                          const MachineBasicBlock *Target) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isMBB() && MO.getMBB() == Target)
      return true;

  return false;
}

/// True if \p MBB contains a branch to \p Target.
static bool hasExplicitBranchTo(const MachineBasicBlock &MBB,
                                const MachineBasicBlock *Target) {
  for (const MachineInstr &MI : MBB.terminators())
    if (MI.isBranch() && branchTargets(MI, Target))
      return true;

  return false;
}

/// Return the unique instruction in \p MBB satisfying \p Pred, or null if there
/// is none or more than one.
template <typename PredT>
static MachineInstr *findUnique(MachineBasicBlock &MBB, PredT Pred) {
  MachineInstr *Result = nullptr;

  for (MachineInstr &MI : MBB) {
    if (!Pred(MI))
      continue;

    if (Result)
      return nullptr;

    Result = &MI;
  }

  return Result;
}

/// True if \p MI is one of the hardware loop markers.
static bool isHWLoopMarker(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoCVHWLoopNoRVCBegin ||
         MI.getOpcode() == RISCV::PseudoCVHWLoopNoRVCEnd;
}

/// True if \p MI is a control instruction for \p Candidate.
static bool isLoopControl(const MachineInstr &MI,
                          const CVHWLoopCandidate &Candidate) {
  return &MI == Candidate.LoopEnd || &MI == Candidate.LoopBranch ||
         &MI == Candidate.UncondBranch;
}

//===----------------------------------------------------------------------===//
// Instruction width helpers
//===----------------------------------------------------------------------===//

/// Return the exact width \p MI will have once the no-RVC region is in force,
/// or nullopt when it is not statically known.
static std::optional<unsigned> getFixedInstrSize(const MachineInstr &MI) {
  if (MI.isDebugInstr() || MI.isMetaInstruction() || isHWLoopMarker(MI))
    return 0u;

  // Length is only an upper bound.
  if (MI.isInlineAsm())
    return std::nullopt;

  // Pseudos are rejected: their expansion is not known to be a single
  // instruction, and the size in the description is not a guarantee.
  if (MI.isPseudo())
    return std::nullopt;

  // Deliberately NOT RISCVInstrInfo::getInstSizeInBytes(): that returns 2 for
  // every instruction the subtarget *could* compress. Inside the no-RVC region
  // those are emitted at full width, so what matters is the description size,
  // which is 2 only for an opcode that is already RVInst16.
  unsigned Size = MI.getDesc().getSize();
  if (Size == 0)
    return std::nullopt;

  return Size;
}

/// Return the width of \p MI inside a hardware-loop body, or nullopt if \p MI
/// may not appear in one.
static std::optional<unsigned> getBodyInstrSize(const MachineInstr &MI) {
  // Control flow that can leave the body. Only a fallthrough chain of blocks
  // is valid.
  if (MI.isCall() || MI.isReturn() || MI.isIndirectBranch() || MI.isBranch())
    return std::nullopt;

  // Invalid instructions for hardware loops.
  if (MI.getOpcode() == RISCV::MRET || MI.getOpcode() == RISCV::DRET ||
      MI.getOpcode() == RISCV::FENCE || MI.getOpcode() == RISCV::FENCE_I ||
      MI.getOpcode() == RISCV::FENCE_TSO || MI.getOpcode() == RISCV::WFI)
    return std::nullopt;

  std::optional<unsigned> Size = getFixedInstrSize(MI);
  if (!Size)
    return std::nullopt;

  // Zero-width markers and debug values contribute nothing. Anything else must
  // be a full-width instruction: a 16-bit opcode cannot be widened by
  // .option norvc
  if (*Size != 0 && *Size != UncompressedInstrSize)
    return std::nullopt;

  return Size;
}

//===----------------------------------------------------------------------===//
// Body measurement
//===----------------------------------------------------------------------===//

/// Walk the body in layout order, accumulating its size and the distance from
/// any already-converted inner loop's end marker to the end of the body.
///
/// The hardware executes exactly [lpstart, lpend) regardless of the CFG, so the
/// loop's blocks must be precisely the blocks laid out between the header and
/// the latch. Both inclusions are checked: every visited block must belong to
/// the loop, and every block of the loop must have been visited.
static std::optional<CVHWLoopBody>
measureBody(const CVHWLoopCandidate &Candidate) {
  CVHWLoopBody Body;
  unsigned VisitedBlocks = 0;

  for (MachineBasicBlock *MBB = Candidate.Header; MBB;
       MBB = MBB->getNextNode()) {
    if (!Candidate.Blocks.contains(MBB))
      return std::nullopt;

    if (!Candidate.Header->sameSection(MBB)) {
      LLVM_DEBUG(dbgs() << "  " << printMBBReference(*MBB)
                        << " is in a different section\n");
      return std::nullopt;
    }

    // A block alignment emits padding at the block label, and that padding is
    // not part of the instruction widths summed below: on the header it could
    // push lpstart past PC + 4, and on any later block it could push the exit
    // label past the lpend computed for it.
    if (MBB->getAlignment() > Align(UncompressedInstrSize)) {
      LLVM_DEBUG(dbgs() << "  " << printMBBReference(*MBB)
                        << " requests alignment stronger than four bytes\n");
      return std::nullopt;
    }
      
    ++VisitedBlocks;

    for (const MachineInstr &MI : *MBB) {
      if (isLoopControl(MI, Candidate))
        continue;

      // End of a nested hardware loop. Reset so that the innermost marker
      // preceding the latch is the one measured against.
      if (MI.getOpcode() == RISCV::PseudoCVHWLoopNoRVCEnd) {
        Body.SawInnerEnd = true;
        Body.BytesAfterInnerEnd = 0;
        continue;
      }

      std::optional<unsigned> Size = getBodyInstrSize(MI);
      if (!Size) {
        LLVM_DEBUG(dbgs() << "  rejected body instruction: " << MI);
        return std::nullopt;
      }

      Body.SizeInBytes += *Size;
      if (Body.SawInnerEnd)
        Body.BytesAfterInnerEnd += *Size;
      if (*Size)
        ++Body.NumInstructions;
    }

    if (MBB == Candidate.Latch) {
      if (VisitedBlocks != Candidate.Blocks.size()) {
        LLVM_DEBUG(dbgs() << "  loop has blocks outside the header/latch range\n");
        return std::nullopt;
      }

      // The exit label is lpend, so padding in front of it moves lpend as well.
      if (Candidate.Exit->getAlignment() > Align(UncompressedInstrSize)) {
        LLVM_DEBUG(dbgs() << "  exit block requests alignment stronger than "
                             "four bytes\n");
        return std::nullopt;
      }

      return Body;
    }
  }

  // Reaching the end of the function without finding the latch.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Trip count
//===----------------------------------------------------------------------===//

static std::optional<CVHWLoopCount> getLoopCount(const MachineInstr &Setup) {
  CVHWLoopCount Count;

  switch (Setup.getOpcode()) {
  case RISCV::PseudoCVHWLoopSetupImm: {
    assert(Setup.getNumExplicitOperands() >= 1 && Setup.getOperand(0).isImm() &&
           "PseudoCVHWLoopSetupImm must take an immediate trip count");

    int64_t Value = Setup.getOperand(0).getImm();

    // lpcount <= 0 leaves the loop inactive, so a hardware loop cannot express
    // it. This should be false by construction, but it is safer to check.
    if (Value <= 0)
      return std::nullopt;

    Count.IsImmediate = true;
    Count.Immediate = static_cast<uint64_t>(Value);
    return Count;
  }
  case RISCV::PseudoCVHWLoopSetup: {
    assert(Setup.getNumExplicitOperands() >= 1 && Setup.getOperand(0).isReg() &&
           "PseudoCVHWLoopSetup must take a register trip count");

    Count.Reg = Setup.getOperand(0).getReg();
    return Count;
  }
  default:
    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// Setup point preconditions and encoding selection
//===----------------------------------------------------------------------===//

/// True when a forward distance of \p Bytes is encodable in a \p OffsetBits wide
/// start/end operand.
static bool isEncodableWithOffset(uint64_t Bytes, unsigned OffsetBits) {
  if (Bytes % CVEndOffsetScale != 0)
    return false;

  return isUIntN(OffsetBits, Bytes / CVEndOffsetScale);
}

/// The setup sequence is emitted at the end of the preheader, which may be
/// later than the position of the setup pseudo. Reading the trip count there is
/// only valid if nothing in between redefines it.
static bool isCountClobberedAtSetupPoint(const CVHWLoopCandidate &Candidate,
                                         const CVHWLoopCount &Count,
                                         const TargetRegisterInfo &TRI) {
  if (Count.IsImmediate)
    return false;

  MachineBasicBlock::iterator I = std::next(Candidate.Setup->getIterator());
  MachineBasicBlock::iterator E = Candidate.Preheader->getFirstTerminator();

  for (; I != E; ++I)
    if (I->modifiesRegister(Count.Reg, &TRI))
      return true;

  return false;
}

/// cv.setup[i]: lpstart is PC + 4 and lpend is the exit label.
static bool canUseCombinedSetup(const CVHWLoopBody &Body,
                                const CVHWLoopCount &Count) {
  if (ForceLongSetup)
    return false;

  return isEncodableWithOffset(CVSetupToHeader + Body.SizeInBytes,
                               Count.IsImmediate ? CVSetupImmEndOffsetBits
                                                 : CVSetupRegEndOffsetBits);
}

/// cv.starti + cv.endi + cv.count[i]. The start offset is a constant checked by
/// the static_assert above; only the end offset depends on the body.
static bool canUseLongSetup(const CVHWLoopBody &Body) {
  return isEncodableWithOffset(CVEndiToHeader + Body.SizeInBytes, CVEndiOffsetBits);
}

//===----------------------------------------------------------------------===//
// Layout
//===----------------------------------------------------------------------===//

/// Once the original backedge is removed, the latch must fall through to the exit
/// block, so the exit block has to be movable to that position.
static bool canPlaceExitAfterLatch(const CVHWLoopCandidate &Candidate) {
  MachineBasicBlock *Latch = Candidate.Latch;
  MachineBasicBlock *Exit = Candidate.Exit;

  // Exit is inserted directly after Latch. Header and Exit are both known
  // successors, so any third successor would be reached by a fallthrough.
  if (Latch->succ_size() != 2)
    return false;

  if (Latch->getNextNode() == Exit)
    return true;

  // Basic block sections and function splitting: the two blocks must be
  // reorderable relative to each other.
  return !Exit->isEHPad() && Latch->sameSection(Exit) &&
         !Exit->isBeginSection() && !Exit->isEndSection() &&
         !Latch->isEndSection();
}

/// Move the exit block to immediately after the latch, making explicit any
/// fallthrough the move breaks.
static void placeExitAfterLatch(CVHWLoopCandidate &Candidate,
                                const RISCVInstrInfo &TII) {
  MachineBasicBlock *Latch = Candidate.Latch;
  MachineBasicBlock *Exit = Candidate.Exit;

  if (Latch->getNextNode() == Exit)
    return;

  MachineBasicBlock *OldPrev = Exit->getPrevNode();
  MachineBasicBlock *OldNext = Exit->getNextNode();

  const bool FixOldPrev = OldPrev && OldPrev != Latch &&
                          OldPrev->isSuccessor(Exit) &&
                          !hasExplicitBranchTo(*OldPrev, Exit);
  const bool FixExit = OldNext && Exit->isSuccessor(OldNext) &&
                       !hasExplicitBranchTo(*Exit, OldNext);

  Exit->moveAfter(Latch);

  if (FixOldPrev)
    TII.insertUnconditionalBranch(*OldPrev, Exit, DebugLoc());
  if (FixExit)
    TII.insertUnconditionalBranch(*Exit, OldNext, DebugLoc());

  // A Latch -> Header fallthrough may have been broken here; the caller removes
  // that edge immediately afterwards.
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Append nops to reach the minimum body length. They go at the end of the
/// latch, before its terminators, so that lpend still falls at the exit label.
static void padBody(const CVHWLoopCandidate &Candidate, unsigned PadCount,
                    const DebugLoc &DL, const RISCVInstrInfo &TII) {
  MachineBasicBlock &Latch = *Candidate.Latch;
  MachineBasicBlock::iterator InsertPt = Latch.getFirstTerminator();

  for (unsigned I = 0; I != PadCount; ++I)
    BuildMI(Latch, InsertPt, DL, TII.get(RISCV::ADDI))
        .addReg(RISCV::X0, RegState::Define)
        .addReg(RISCV::X0)
        .addImm(0);
}

/// Open the no-RVC region and return the point at which the setup sequence is
/// emitted: after every other instruction in the preheader, and therefore
/// directly before the header.
static MachineBasicBlock::iterator
openSetupPoint(CVHWLoopCandidate &Candidate, const DebugLoc &DL,
               const RISCVInstrInfo &TII) {
  MachineBasicBlock &Preheader = *Candidate.Preheader;
  MachineBasicBlock::iterator InsertPt = Preheader.getFirstTerminator();

  BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::PseudoCVHWLoopNoRVCBegin));

  return InsertPt;
}

/// Erase the branch to the header, which the header's placement has made
/// redundant. Nothing may sit between the setup sequence and lpstart.
static void closeSetupPoint(CVHWLoopCandidate &Candidate,
                            MachineBasicBlock::iterator InsertPt) {
  MachineBasicBlock &Preheader = *Candidate.Preheader;

  if (InsertPt == Preheader.end())
    return;

  assert(InsertPt->isUnconditionalBranch() &&
         branchTargets(*InsertPt, Candidate.Header) &&
         "unexpected preheader terminator");

  Preheader.erase(InsertPt);
}

/// Emit cv.setup/cv.setupi as the last instruction of the preheader.
static void emitCombinedSetup(CVHWLoopCandidate &Candidate,
                              const CVHWLoopCount &Count, unsigned LoopID,
                              const DebugLoc &DL, const RISCVInstrInfo &TII) {
  MachineBasicBlock &Preheader = *Candidate.Preheader;
  MachineBasicBlock::iterator InsertPt = openSetupPoint(Candidate, DL, TII);

  if (Count.IsImmediate)
    BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_SETUPI))
        .addImm(LoopID)
        .addImm(Count.Immediate)
        .addMBB(Candidate.Exit);
  else
    BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_SETUP))
        .addImm(LoopID)
        .addReg(Count.Reg)
        .addMBB(Candidate.Exit);

  Candidate.Exit->setLabelMustBeEmitted();

  closeSetupPoint(Candidate, InsertPt);
}

/// Emit cv.starti + cv.endi + cv.count/cv.counti, used when the loop end does
/// not fit the five-bit field of cv.setupi/cv.setupi.
static void emitLongSetup(CVHWLoopCandidate &Candidate,
                          const CVHWLoopCount &Count, unsigned LoopID,
                          const DebugLoc &DL, const RISCVInstrInfo &TII) {
  MachineBasicBlock &Preheader = *Candidate.Preheader;
  MachineBasicBlock::iterator InsertPt = openSetupPoint(Candidate, DL, TII);

  BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_STARTI))
      .addImm(LoopID)
      .addMBB(Candidate.Header);
  BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_ENDI))
      .addImm(LoopID)
      .addMBB(Candidate.Exit);

  if (Count.IsImmediate)
    BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_COUNTI))
        .addImm(LoopID)
        .addImm(Count.Immediate);
  else
    BuildMI(Preheader, InsertPt, DL, TII.get(RISCV::CV_COUNT))
        .addImm(LoopID)
        .addReg(Count.Reg);

  Candidate.Header->setLabelMustBeEmitted();
  Candidate.Exit->setLabelMustBeEmitted();

  closeSetupPoint(Candidate, InsertPt);
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// Classify the latch's terminators. A hardware-loop latch is either
///
///   conditional branch to Header, unconditional branch to Exit
///
/// or
///
///   conditional branch to Exit, fallthrough/unconditional branch to Header.
///
/// Return nullopt for an unsupported sequence, nullptr when there is no
/// unconditional branch, and the branch otherwise.
static std::optional<MachineInstr *>
findLoopUncondBranchBranch(const CVHWLoopCandidate &Candidate) {
  MachineInstr *UncondBranch = nullptr;

  for (MachineInstr &MI : Candidate.Latch->terminators()) {
    if (&MI == Candidate.LoopBranch)
      continue;

    if (UncondBranch || !MI.isUnconditionalBranch() ||
        (!branchTargets(MI, Candidate.Header) &&
         !branchTargets(MI, Candidate.Exit)))
      return std::nullopt;

    UncondBranch = &MI;
  }

  return UncondBranch;
}

/// Performs loop analysis.
std::optional<CVHWLoopCandidate>
RISCVHardwareLoops::analyzeLoop(MachineLoop &ML) const {
  CVHWLoopCandidate Candidate;

  Candidate.Loop = &ML;
  Candidate.Preheader = ML.getLoopPreheader();
  Candidate.Header = ML.getHeader();
  Candidate.Latch = ML.getLoopLatch();
  Candidate.Exit = ML.getExitBlock();

  if (!Candidate.Preheader || !Candidate.Header || !Candidate.Latch ||
      !Candidate.Exit)
    return std::nullopt;

  // getExitBlock() gives the loop's unique exit, but the latch must also be the
  // exiting block for lpend to be meaningful.
  if (!Candidate.Latch->isSuccessor(Candidate.Exit))
    return std::nullopt;

  Candidate.Setup = findUnique(*Candidate.Preheader, [](const MachineInstr &MI) {
    return MI.getOpcode() == RISCV::PseudoCVHWLoopSetup ||
           MI.getOpcode() == RISCV::PseudoCVHWLoopSetupImm;
  });
  Candidate.LoopEnd = findUnique(*Candidate.Latch, [](const MachineInstr &MI) {
    return MI.getOpcode() == RISCV::PseudoCVHWLoopEnd;
  });

  if (!Candidate.Setup || !Candidate.LoopEnd)
    return std::nullopt;

  assert(Candidate.LoopEnd->getNumExplicitDefs() == 1 &&
         Candidate.LoopEnd->getOperand(0).isReg() &&
         "PseudoCVHWLoopEnd must define exactly one register");

  // The loop branch is the unique latch terminator consuming the continue flag.
  // A branch that consumes it but targets neither the header nor the exit is
  // skipped here and rejected below, since it cannot be a valid UncondBranch.
  Register ContinueReg = Candidate.LoopEnd->getOperand(0).getReg();
  Candidate.LoopBranch =
      findUnique(*Candidate.Latch, [&](const MachineInstr &MI) {
        return MI.isBranch() && MI.isTerminator() &&
               MI.readsRegister(ContinueReg, TRI) &&
               (branchTargets(MI, Candidate.Header) ||
                branchTargets(MI, Candidate.Exit));
      });

  if (!Candidate.LoopBranch)
    return std::nullopt;

  std::optional<MachineInstr *> UncondBranch = findLoopUncondBranchBranch(Candidate);
  if (!UncondBranch)
    return std::nullopt;

  Candidate.UncondBranch = *UncondBranch;

  for (MachineBasicBlock *MBB : ML.blocks())
    Candidate.Blocks.insert(MBB);

  return Candidate;
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

bool RISCVHardwareLoops::convertCandidate(CVHWLoopCandidate &Candidate,
                                          unsigned LoopID, bool IsOuter) {
  assert(LoopID < 2 && "CV32E40P provides two hardware-loop register sets");

  LLVM_DEBUG(dbgs() << "Considering " << printMBBReference(*Candidate.Header)
                    << " for hardware loop " << LoopID << '\n');

  std::optional<CVHWLoopBody> Body = measureBody(Candidate);
  if (!Body) {
    LLVM_DEBUG(dbgs() << "  body is not a legal hardware loop body\n");
    return false;
  }

  unsigned PadCount = 0;
  if (Body->NumInstructions < CVMinBodyInstructions) {
    // A body below the minimum can be padded with nops, at the cost of one
    // wasted cycle per iteration for each.
    if (!PadShortBodies) {
      LLVM_DEBUG(dbgs() << "  body has fewer than " << CVMinBodyInstructions
                        << " instructions\n");
      return false;
    }

    PadCount = CVMinBodyInstructions - Body->NumInstructions;
    Body->NumInstructions += PadCount;
    Body->SizeInBytes += PadCount * UncompressedInstrSize;
    Body->BytesAfterInnerEnd += PadCount * UncompressedInstrSize;
  }

  // An outer loop can only be converted once its inner loop has been, since the
  // separation is measured against the inner loop's end marker.
  if (IsOuter && (!Body->SawInnerEnd ||
                  Body->BytesAfterInnerEnd < CVMinNestedEndSeparation)) {
    LLVM_DEBUG(dbgs() << "  insufficient separation from the inner loop end\n");
    return false;
  }

  std::optional<CVHWLoopCount> Count = getLoopCount(*Candidate.Setup);
  if (!Count) {
    LLVM_DEBUG(dbgs() << "  unusable trip count\n");
    return false;
  }

  if (isCountClobberedAtSetupPoint(Candidate, *Count, *TRI)) {
    LLVM_DEBUG(dbgs() << "  trip count is clobbered before the setup point\n");
    return false;
  }

  if (!canPlaceExitAfterLatch(Candidate)) {
    LLVM_DEBUG(dbgs() << "  exit block cannot follow the latch\n");
    return false;
  }

  // The preheader successor can only change when the exit block currently
  // follows it.
  const bool ExitWillMove = Candidate.Latch->getNextNode() != Candidate.Exit;
  const bool HeaderWillFollowPreheader =
      Candidate.Preheader->getNextNode() == Candidate.Header ||
      (ExitWillMove && Candidate.Preheader->getNextNode() == Candidate.Exit &&
       Candidate.Exit->getNextNode() == Candidate.Header);

  if (!HeaderWillFollowPreheader) {
    LLVM_DEBUG(dbgs() << "  header would not directly follow the preheader\n");
    return false;
  }

  const bool UseCombinedSetup = canUseCombinedSetup(*Body, *Count);
  if (!UseCombinedSetup && !canUseLongSetup(*Body)) {
    LLVM_DEBUG(dbgs() << "  loop end offset does not fit any encoding\n");
    return false;
  }

  // Every check has passed. The function is modified from here on.
  DebugLoc DL = Candidate.Setup->getDebugLoc();

  placeExitAfterLatch(Candidate, *TII);
  assert(Candidate.Preheader->getNextNode() == Candidate.Header &&
         "adjacency prediction disagreed with the resulting layout");

  // Pad the loop bpdy (if needed).
  padBody(Candidate, PadCount, DL, *TII);

  if (UseCombinedSetup)
    emitCombinedSetup(Candidate, *Count, LoopID, DL, *TII);
  else
    emitLongSetup(Candidate, *Count, LoopID, DL, *TII);

  BuildMI(*Candidate.Exit, Candidate.Exit->getFirstNonPHI(), DL,
          TII->get(RISCV::PseudoCVHWLoopNoRVCEnd));

  if (Candidate.UncondBranch)
    Candidate.UncondBranch->eraseFromParent();
  Candidate.LoopBranch->eraseFromParent();
  Candidate.LoopEnd->eraseFromParent();
  Candidate.Setup->eraseFromParent();

  Candidate.Latch->removeSuccessor(Candidate.Header,
                                   /*NormalizeSuccProbs=*/ true);

  LLVM_DEBUG(dbgs() << "  converted using "
                    << (UseCombinedSetup ? "cv.setup" : "cv.starti/cv.endi")
                    << ", body " << Body->SizeInBytes << " bytes\n");
  ++NumHardwareLoops;
  return true;
}

bool RISCVHardwareLoops::processLoopTree(MachineLoop &ML) {
  ArrayRef<MachineLoop *> SubLoops = ML.getSubLoops();

  // An innermost loop uses hardware-loop register set 0.
  if (SubLoops.empty()) {
    std::optional<CVHWLoopCandidate> Candidate = analyzeLoop(ML);
    return Candidate && convertCandidate(*Candidate, /*LoopID=*/ 0,
                                         /*IsOuter=*/ false);
  }

  // A loop whose children are all innermost can take register set 1, with every
  // child taking set 0. The children are disjoint and run in sequence, so they
  // can share the set: each re-arms it on entry and it lies dormant in between.
  const bool AllChildrenInnermost =
      all_of(SubLoops, [](const MachineLoop *SubLoop) {
        return SubLoop->getSubLoops().empty();
      });

  if (AllChildrenInnermost) {
    // Everything is analyzed before any conversion runs, because analyzeLoop()
    // reads MachineLoopInfo and the first backedge removal leaves it stale.

    std::optional<CVHWLoopCandidate> OuterCandidate = analyzeLoop(ML);

    SmallVector<std::optional<CVHWLoopCandidate>, 4> InnerCandidates;
    for (MachineLoop *SubLoop : SubLoops)
      InnerCandidates.push_back(analyzeLoop(*SubLoop));

    bool Changed = false;
    for (std::optional<CVHWLoopCandidate> &InnerCandidate : InnerCandidates)
      if (InnerCandidate)
        Changed |= convertCandidate(*InnerCandidate, /*LoopID=*/ 0,
                                    /*IsOuter=*/ false);

    if (!Changed)
      return false;

    if (OuterCandidate &&
        !convertCandidate(*OuterCandidate, /*LoopID=*/ 1, /*IsOuter=*/ true))
      LLVM_DEBUG(dbgs() << "  parent loop stays a software loop\n");

    return true;
  }

  bool Changed = false;
  for (MachineLoop *SubLoop : SubLoops)
    Changed |= processLoopTree(*SubLoop);

  return Changed;
}

bool RISCVHardwareLoops::runOnMachineFunction(MachineFunction &MF) {
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVendorXCVhwlp())
    return false;

  if (DisableHardwareLoops || skipFunction(MF.getFunction()))
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  // Snapshot the top-level loops: MachineLoopInfo is required but deliberately
  // not preserved, and becomes stale as soon as a backedge is removed.
  SmallVector<MachineLoop *, 8> TopLevelLoops(MLI.begin(), MLI.end());

  bool Changed = false;
  for (MachineLoop *TopLevelLoop : TopLevelLoops)
    Changed |= processLoopTree(*TopLevelLoop);

  if (Changed)
    MF.RenumberBlocks();

  return Changed;
}
