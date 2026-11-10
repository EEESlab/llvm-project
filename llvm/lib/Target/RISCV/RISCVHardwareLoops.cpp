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
// Linker relaxation is left enabled inside the loop: if it shrinks the body,
// the loop end offset is fixed up by the R_RISCV_CVPCREL_UI12/URS1 relocation
// that the assembler emits on the setup instruction.
//
// The matching .option pop is emitted at the top of the exit block, which is
// moved to directly follow the latch. If the exit block is already pinned in
// place by a hardware loop converted earlier (it is that loop's lpend, or its
// preheader), it is not moved; instead a new block holding only a jump to the
// exit is inserted after the latch and becomes this loop's lpend.
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
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
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

/// CV32E40P requires the outer loop end address to be at least one instruction
/// between the inner and outer hardware loops.
constexpr unsigned CVMinNestedEndSeparation = 1;

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

  /// The software counter LSR left in the body, dead once the loop control is
  /// gone. Excluded from the body measurement and erased on conversion.
  SmallVector<MachineInstr *, 2> DeadCounter;

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
  /// Upper bound on the instructions the linker could delete due to relaxations.
  uint64_t RelaxableInstructions = 0;
  /// Set when an already-converted inner loop's end marker was seen.
  bool SawInnerEnd = false;
  /// Instructions between the last inner end marker and the end of the body.
  uint64_t InstructionsAfterInnerEnd = 0;
  /// How many of those the linker could delete.
  uint64_t RelaxableAfterInnerEnd = 0;
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
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const RISCVInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MachineBlockFrequencyInfo *MBFI = nullptr;

  /// Order sibling loops so that the most frequently entered one is converted
  /// first.
  SmallVector<MachineLoop *, 4> sortByEntryFrequency(
      ArrayRef<MachineLoop *> Loops) const;

  /// Blocks whose position is fixed by a hardware loop converted earlier in
  /// this function: each loop's preheader (its setup must stay directly before
  /// the header) and its exit block (its label is lpend, so it must stay
  /// directly after the latch). Moving one of them would break that loop.
  SmallPtrSet<const MachineBasicBlock *, 16> PinnedBlocks;

  std::optional<CVHWLoopCandidate> analyzeLoop(MachineLoop &ML) const;

  bool processLoopTree(MachineLoop &ML);

  bool convertCandidate(CVHWLoopCandidate &Candidate, unsigned LoopID,
                        bool IsOuter);

  bool tryConvertCandidate(CVHWLoopCandidate &Candidate, unsigned LoopID,
                           bool IsOuter);

  bool foldDuplicateCounter(MachineInstr &LoopEnd) const;
};

} // end anonymous namespace

char RISCVHardwareLoops::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVHardwareLoops, DEBUG_TYPE,
                      "RISC-V CORE-V hardware loops", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
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
  if (&MI == Candidate.LoopEnd || &MI == Candidate.LoopBranch ||
      &MI == Candidate.UncondBranch)
    return true;

  return is_contained(Candidate.DeadCounter, &MI);
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
  // The exceptions below are expanded, after this pass, into exactly one
  // 32-bit instruction with no compressed form, so they are four bytes inside
  // the no-RVC region. Control-flow pseudos with a single-instruction
  // expansion (PseudoBR, PseudoRET, ...) are not listed: getBodyInstrSize
  // rejects branches, calls and returns before asking for a size. If any
  // expansion below changes, this list must change with it.
  if (MI.isPseudo()) {
    switch (MI.getOpcode()) {
    // Zfinx/Zhinx register moves, e.g. the zeroing of a float accumulator
    // before an inner loop. RISCVExpandPseudo: ADDI rd, rs, 0.
    case RISCV::PseudoMV_FPR16INX:
    case RISCV::PseudoMV_FPR32INX:
    // Xcvsimd byte shuffle with an 8-bit immediate. RISCVExpandPseudo picks
    // one of cv.shufflei{0,1,2,3}.sci.b from bits [7:6].
    case RISCV::CV_SHUFFLE_SCI_B_PSEUDO:
    // Floating-point CSR accesses, e.g. a rounding-mode change or a strict-FP
    // quiet comparison inside a loop. Lowered 1:1 by PseudoInstExpansion to
    // csrrs/csrrw/csrrwi/csrrc/csrrci/csrrsi.
    case RISCV::ReadFRM:
    case RISCV::WriteFRM:
    case RISCV::WriteFRMImm:
    case RISCV::SwapFRMImm:
    case RISCV::ReadFFLAGS:
    case RISCV::WriteFFLAGS:
    case RISCV::ReadFCSR:
    case RISCV::WriteFCSR:
    case RISCV::WriteFCSRImm:
    case RISCV::ClearFCSR:
    case RISCV::ClearFCSRImm:
    case RISCV::SetFCSR:
    case RISCV::SetFCSRImm:
      return UncompressedInstrSize;
    default:
      return std::nullopt;
    }
  }

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

/// Return how many instruction the linker could delete from \p MI's relaxation
/// sequence, counting only the instruction that anchors it so that a sequence
/// is not counted more than once.
///
/// This is an upper bound: whether a given symbol is in range of gp, x0 or the
/// thread pointer is a link-time property.
static unsigned getRelaxableInstructions(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    switch (MO.getTargetFlags()) {
    // lui %hi(sym) + l/s %lo(sym), or auipc %pcrel_hi + l/s %pcrel_lo.
    // Folding to gp or x0 removes the high half.
    case RISCVII::MO_LO:
    case RISCVII::MO_PCREL_LO:
      return 1;

    // lui %tprel_hi + add %tprel_add + l/s %tprel_lo. Relaxing to LE removes
    // the lui and the add.
    case RISCVII::MO_TPREL_LO:
      return 2;

    // auipc + l[dw] + addi + jalr. Relaxing to IE or LE removes the first two.
    case RISCVII::MO_TLSDESC_CALL:
      return 2;

    default:
      break;
    }
  }

  return 0;
}

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
        Body.InstructionsAfterInnerEnd = 0;
        continue;
      }

      std::optional<unsigned> Size = getBodyInstrSize(MI);
      if (!Size) {
        LLVM_DEBUG(dbgs() << "  rejected body instruction: " << MI);
        return std::nullopt;
      }

      Body.SizeInBytes += *Size;

      unsigned RelaxableInsns = getRelaxableInstructions(MI);
      Body.RelaxableInstructions += RelaxableInsns;

      if (Body.SawInnerEnd) {
        ++Body.InstructionsAfterInnerEnd;
        Body.RelaxableAfterInnerEnd += RelaxableInsns;
      }

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
    assert(Setup.getNumExplicitOperands() == 1 && Setup.getOperand(0).isImm() &&
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
    assert(Setup.getNumExplicitOperands() == 1 && Setup.getOperand(0).isReg() &&
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

  LLVM_DEBUG(dbgs() << "    count reg: " << printReg(Count.Reg, &TRI) << "\n");
  for (; I != E; ++I)
    // The counter initialization is erased on conversion.
    if (!is_contained(Candidate.DeadCounter, &*I) &&
        I->modifiesRegister(Count.Reg, &TRI)) {
      LLVM_DEBUG(dbgs() << "    clobbered by: " << *I);
      return true;
    }

  return false;
}

/// The trip count register can be reused after the setup pseudo while a copy
/// of it is still intact at the setup point, typically the move that starts
/// LSR's countdown:
///
///     PseudoCVHWLoopSetup $x13
///     $x12 = ADDI killed $x13, 0    ; countdown start
///     $x13 = LW %stack.2            ; $x13 reused
///
/// Return a register that holds the trip count at the end of the preheader.
/// A move that produces it is kept, even if it was collected as part of the
/// dead counter: it now feeds the setup.
static std::optional<Register>
findCountCopyAtSetupPoint(CVHWLoopCandidate &Candidate,
                          const CVHWLoopCount &Count,
                          const TargetRegisterInfo &TRI) {
  if (Count.IsImmediate ||
      Candidate.Setup->getParent() != Candidate.Preheader)
    return std::nullopt;

  // Registers holding the trip count, with the move that produced each one.
  SmallVector<std::pair<Register, MachineInstr *>, 4> Equal;
  Equal.push_back({Count.Reg, nullptr});
  auto Find = [&](Register R) {
    return find_if(Equal, [&](const auto &E) { return E.first == R; });
  };

  for (MachineInstr &MI :
       make_range(std::next(MachineBasicBlock::iterator(Candidate.Setup)),
                  Candidate.Preheader->getFirstTerminator())) {
    const bool IsMove =
        (MI.isCopy() ||
         (MI.getOpcode() == RISCV::ADDI && MI.getOperand(2).isImm() &&
          MI.getOperand(2).getImm() == 0)) &&
        MI.getOperand(0).isReg() && MI.getOperand(1).isReg();
    if (IsMove && Find(MI.getOperand(1).getReg()) != Equal.end()) {
      Register Dst = MI.getOperand(0).getReg();
      auto It = Find(Dst);
      if (It != Equal.end())
        It->second = &MI;
      else
        Equal.push_back({Dst, &MI});
      continue;
    }

    erase_if(Equal, [&](const auto &E) {
      return MI.modifiesRegister(E.first, &TRI);
    });
    if (Equal.empty())
      return std::nullopt;
  }

  if (Equal.empty())
    return std::nullopt;

  auto [Reg, Def] = Equal.front();
  if (Def)
    erase(Candidate.DeadCounter, Def);
  return Reg;
}

/// Collect the software counter LSR built alongside the hardware loop.
///
/// LSR does not know that llvm.loop.decrement.reg returns its argument minus
/// one, so it creates a second countdown on the same register and feeds the
/// loop-carried phi from that instead of from the intrinsic. After conversion
/// the hardware maintains lpcount and the countdown has no consumer.
///
/// It is removable exactly when the count register is dead on leaving the loop
/// and nothing in the body reads it except the countdown itself.
static void findDeadSoftwareCounter(CVHWLoopCandidate &Candidate,
                                    const CVHWLoopCount &Count,
                                    const TargetRegisterInfo &TRI) {
  Candidate.DeadCounter.clear();

  auto IsMove = [](const MachineInstr &MI) {
    return (MI.isCopy() ||
            (MI.getOpcode() == RISCV::ADDI && MI.getNumExplicitOperands() == 3 &&
             MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0)) &&
           MI.getOperand(0).isReg() && MI.getOperand(1).isReg();
  };

  MachineBasicBlock *EndMBB = Candidate.LoopEnd->getParent();
  Register CounterReg = Candidate.LoopEnd->getOperand(1).getReg();
  for (MachineBasicBlock::iterator I = Candidate.LoopEnd->getIterator(),
                                   B = EndMBB->begin();
       I != B;) {
    --I;
    if (I->modifiesRegister(CounterReg, &TRI)) {
      if (IsMove(*I))
        CounterReg = I->getOperand(1).getReg();
      break;
    }
  }

  // The move that initializes the counter in the preheader. The counter may
  // also share its register with the trip count: the setup pseudo consumes
  // the trip count, and a move after it reloads the register with the
  // countdown's start value. That move is part of the counter as well, and
  // left in place it would clobber the trip count before the setup point.
  // A definition before the setup pseudo is the trip count itself and stays.
  // With a constant trip count, the countdown starts from a load immediate.
  // A countdown that was spilled in the parent loop starts from a reload.
  auto IsInit = [&](const MachineInstr &MI) {
    if (!MI.getOperand(0).isReg() || MI.getOperand(0).getReg() != CounterReg)
      return false;
    if (IsMove(MI))
      return true;
    if (MI.getOpcode() == RISCV::LW && MI.hasOneMemOperand()) {
      const MachineMemOperand *MMO = *MI.memoperands_begin();
      if (!MMO->isVolatile() && !MMO->isAtomic() && MMO->getPseudoValue() &&
          isa<FixedStackPseudoSourceValue>(MMO->getPseudoValue()))
        return true;
    }
    return Count.IsImmediate && MI.getOpcode() == RISCV::ADDI &&
           MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == RISCV::X0 &&
           MI.getOperand(2).isImm();
  };

  const bool SharesCountReg = !Count.IsImmediate && CounterReg == Count.Reg;
  MachineInstr *CounterInit = nullptr;
  if (!SharesCountReg || Candidate.Setup->getParent() == Candidate.Preheader)
    for (MachineInstr &MI : reverse(*Candidate.Preheader)) {
      if (SharesCountReg && &MI == Candidate.Setup)
        break;
      if (MI.modifiesRegister(CounterReg, &TRI)) {
        if (IsInit(MI))
          CounterInit = &MI;
        break;
      }
    }

  LLVM_DEBUG(dbgs() << "  counter: " << printReg(CounterReg, &TRI)
                    << " feeds the loop end\n");

  // Anything after the loop may still read the count.
  LivePhysRegs LPR(TRI);
  LPR.addLiveOuts(*Candidate.Exit);
  for (MachineInstr &MI : reverse(*Candidate.Exit))
    LPR.stepBackward(MI);

  if (LPR.contains(CounterReg)) {
    LLVM_DEBUG(dbgs() << "  counter: " << printReg(CounterReg, &TRI)
           << " is live out of " << printMBBReference(*Candidate.Exit) << "\n");
    return;
  }

  for (MachineBasicBlock *MBB : Candidate.Blocks) {
    for (MachineInstr &MI : *MBB) {
      // Already erased by the conversion; not a use of the counter.
      if (isLoopControl(MI, Candidate))
        continue;

      const bool Reads = MI.readsRegister(CounterReg, &TRI);
      const bool Writes = MI.modifiesRegister(CounterReg, &TRI);

      if (!Reads && !Writes)
        continue;

      // A read that is not part of the countdown is a real use.
      if (Reads && !Writes) {
        // A copy feeding the loop end pseudo is part of the counter chain, not
        // a use: the allocator inserts it to satisfy the pseudo's tied operand,
        // and it dies with the pseudo.
        if (MI.isCopy() || MI.getOpcode() == RISCV::ADDI) {
          Register Dst = MI.getOperand(0).getReg();
          if (Candidate.LoopEnd->readsRegister(Dst, &TRI)) {
            Candidate.DeadCounter.push_back(&MI);
            continue;
          }
        }

        LLVM_DEBUG(dbgs() << "  counter: real use: " << MI);
        Candidate.DeadCounter.clear();
        return;
      }

      // Only a plain single-def instruction can be dropped: anything with a
      // second def or a side effect may be doing something else as well.
      if (MI.getNumExplicitDefs() != 1 || MI.mayLoadOrStore() ||
          MI.hasUnmodeledSideEffects()) {
        LLVM_DEBUG(dbgs() << "  counter: unsuitable: " << MI);
        Candidate.DeadCounter.clear();
        return;
      }

      Candidate.DeadCounter.push_back(&MI);
    }
  }

  // The copy that initialized the counter is dead once nothing reads the
  // register it produced, whether or not a countdown was found in the body.
  // Instructions about to be erased are not readers, which is what
  // isLoopControl() covers: PseudoCVHWLoopEnd reads the counter through its
  // tied operand.
  if (CounterInit) {
    bool ReadAnywhere = false;

    if (LPR.contains(CounterReg)) {
      LLVM_DEBUG(dbgs() << "  counter init: " << printReg(CounterReg, &TRI)
                        << " is live out\n");
      ReadAnywhere = true;
    }

    // Only reads after the move see the value it produces.
    for (MachineInstr &MI : make_range(std::next(MachineBasicBlock::iterator(CounterInit)),
                                       Candidate.Preheader->end()))
      if (MI.readsRegister(CounterReg, &TRI)) {
        LLVM_DEBUG(dbgs() << "  counter init: read in preheader by: " << MI);
        ReadAnywhere = true;
      }

    for (MachineBasicBlock *MBB : Candidate.Blocks)
      for (MachineInstr &MI : *MBB)
        if (!isLoopControl(MI, Candidate) &&
            MI.readsRegister(CounterReg, &TRI)) {
          LLVM_DEBUG(dbgs() << "  counter init: read in body by: " << MI);
          ReadAnywhere = true;
        }

    if (!ReadAnywhere)
      Candidate.DeadCounter.push_back(CounterInit);
  }

  LLVM_DEBUG(dbgs() << "  counter: " << Candidate.DeadCounter.size()
         << " instruction(s) to remove\n");
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

/// cv.starti + cv.endi + cv.count[i].
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

/// Check if a new block can be inserted directly after the latch.
static bool canInsertExitTrampoline(const CVHWLoopCandidate &Candidate) {
  const MachineBasicBlock *Latch = Candidate.Latch;

  // Same idea as canPlaceExitAfterLatch(): Header and Exit are the only
  // successors, and the new block is placed in the latch's section.
  return Latch->succ_size() == 2 && !Latch->isEndSection();
}

/// Insert, directly after the latch, a block that jumps to the real exit,
/// and redirect the latch's exit edge to it.
static MachineBasicBlock *insertExitTrampoline(CVHWLoopCandidate &Candidate,
                                               const RISCVInstrInfo &TII) {
  MachineBasicBlock *Latch = Candidate.Latch;
  MachineBasicBlock *Exit = Candidate.Exit;
  MachineFunction &MF = *Latch->getParent();

  MachineBasicBlock *Trampoline =
      MF.CreateMachineBasicBlock(Exit->getBasicBlock());
  MF.insert(std::next(Latch->getIterator()), Trampoline);
  Trampoline->setSectionID(Latch->getSectionID());

  // Everything live into the exit is live through the trampoline.
  for (const MachineBasicBlock::RegisterMaskPair &LI : Exit->liveins())
    Trampoline->addLiveIn(LI);
  Trampoline->sortUniqueLiveIns();

  // The latch's branches to Exit are erased by the caller, so only the CFG
  // edge needs updating here.
  Latch->replaceSuccessor(Exit, Trampoline);
  Trampoline->addSuccessor(Exit);
  TII.insertUnconditionalBranch(*Trampoline, Exit, DebugLoc());

  return Trampoline;
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

/// Emit cv.starti + cv.endi + cv.count/cv.counti.
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
findLoopUncondBranch(const CVHWLoopCandidate &Candidate) {
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

  Candidate.Preheader = ML.getLoopPreheader();
  Candidate.Header = ML.getHeader();
  Candidate.Latch = ML.getLoopLatch();
  Candidate.Exit = ML.getExitBlock();

  if (!Candidate.Preheader || !Candidate.Header || !Candidate.Latch ||
      !Candidate.Exit) {
    LLVM_DEBUG(dbgs() << "Skipping " << printMBBReference(*ML.getHeader())
                      << ": no preheader, no single latch, or no unique exit\n");
    return std::nullopt;
  }

  // getExitBlock() gives the loop's unique exit, but the latch must also be the
  // exiting block for lpend to be meaningful.
  if (!Candidate.Latch->isSuccessor(Candidate.Exit)) {
    LLVM_DEBUG(dbgs() << "Skipping " << printMBBReference(*Candidate.Header)
                      << ": the latch is not the exiting block\n");
    return std::nullopt;
  }

  Candidate.Setup = findUnique(*Candidate.Preheader, [](const MachineInstr &MI) {
    return MI.getOpcode() == RISCV::PseudoCVHWLoopSetup ||
           MI.getOpcode() == RISCV::PseudoCVHWLoopSetupImm;
  });
  Candidate.LoopEnd = findUnique(*Candidate.Latch, [](const MachineInstr &MI) {
    return MI.getOpcode() == RISCV::PseudoCVHWLoopEnd;
  });

  if (!Candidate.Setup || !Candidate.LoopEnd) {
    LLVM_DEBUG(dbgs() << "Skipping " << printMBBReference(*Candidate.Header)
                      << ": no unique setup pseudo in the preheader or end "
                         "pseudo in the latch\n");
    return std::nullopt;
  }

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

  if (!Candidate.LoopBranch) {
    LLVM_DEBUG(dbgs() << "Skipping " << printMBBReference(*Candidate.Header)
                      << ": no unique latch branch reading the continue flag\n");
    return std::nullopt;
  }

  std::optional<MachineInstr *> UncondBranch = findLoopUncondBranch(Candidate);
  if (!UncondBranch) {
    LLVM_DEBUG({
      dbgs() << "Skipping " << printMBBReference(*Candidate.Header)
             << ": unsupported second terminator in the latch\n";
      for (MachineInstr &MI : Candidate.Latch->terminators())
        dbgs() << "    terminator: " << MI
               << "      uncond=" << MI.isUnconditionalBranch()
               << " toHeader=" << branchTargets(MI, Candidate.Header)
               << " toExit=" << branchTargets(MI, Candidate.Exit) << "\n";
      dbgs() << "    exit=" << printMBBReference(*Candidate.Exit) << "\n";
    });
    return std::nullopt;
  }

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
  return tryConvertCandidate(Candidate, LoopID, IsOuter);
}

/// A loop that is not converted keeps PseudoCVHWLoopEnd, which is later
/// expanded to "addi Tmp, Tmp, -1". Next to it sits the countdown LSR created
/// (see findDeadSoftwareCounter()), so the latch ends up as
///
///     Tmp = COPY Counter            ; feeds the pseudo's tied operand
///     Tmp = PseudoCVHWLoopEnd Tmp   ; -> addi Tmp, Tmp, -1
///     Counter = ADDI Counter, -1    ; LSR's countdown
///     branch on Tmp
///
/// Both registers hold the same value. Rewrite the pseudo and the branch to
/// use Counter directly and erase the copy and LSR's decrement:
///
///     Counter = PseudoCVHWLoopEnd Counter
///     branch on Counter
///
/// This only looks at the latch, so it also applies to loops that were never
/// conversion candidates, e.g. because the setup pseudo is not in the
/// preheader.
bool RISCVHardwareLoops::foldDuplicateCounter(MachineInstr &LoopEnd) const {
  MachineBasicBlock &Latch = *LoopEnd.getParent();

  if (!LoopEnd.getOperand(0).isReg() || !LoopEnd.getOperand(1).isReg())
    return false;
  const Register Tmp = LoopEnd.getOperand(0).getReg();
  if (LoopEnd.getOperand(1).getReg() != Tmp)
    return false;

  // The move into Tmp feeding the pseudo, with nothing else touching Tmp, and
  // the source unchanged, in between.
  MachineInstr *Copy = nullptr;
  for (MachineBasicBlock::iterator I = LoopEnd.getIterator(), B = Latch.begin();
       I != B;) {
    --I;
    if (I->modifiesRegister(Tmp, TRI)) {
      const bool IsMove =
          I->isCopy() ||
          (I->getOpcode() == RISCV::ADDI && I->getOperand(1).isReg() &&
           I->getOperand(2).isImm() && I->getOperand(2).getImm() == 0);
      if (IsMove && I->getOperand(0).getReg() == Tmp &&
          I->getOperand(1).isReg() && I->getOperand(1).getReg() != Tmp &&
          I->getOperand(1).getReg() != RISCV::X0)
        Copy = &*I;
      break;
    }
    if (I->readsRegister(Tmp, TRI))
      return false;
  }
  if (!Copy)
    return false;

  const Register Counter = Copy->getOperand(1).getReg();
  for (MachineInstr &MI :
       make_range(std::next(MachineBasicBlock::iterator(Copy)),
                  MachineBasicBlock::iterator(LoopEnd)))
    if (MI.modifiesRegister(Counter, TRI))
      return false;

  // LSR's decrement after the pseudo. In between, Counter must not be read
  // or written: after the fold it already holds the decremented value there.
  // Tmp must only be read by the loop branch.
  MachineInstr *Decrement = nullptr;
  MachineInstr *Branch = nullptr;
  for (MachineInstr &MI :
       make_range(std::next(MachineBasicBlock::iterator(LoopEnd)), Latch.end())) {
    if (MI.readsRegister(Tmp, TRI)) {
      if (Branch || !MI.isConditionalBranch())
        return false;
      Branch = &MI;
    }
    if (MI.modifiesRegister(Tmp, TRI))
      return false;

    if (Decrement)
      continue;
    if (MI.getOpcode() == RISCV::ADDI && MI.getOperand(0).getReg() == Counter &&
        MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Counter &&
        MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == -1) {
      if (Branch)
        return false;
      Decrement = &MI;
      continue;
    }
    if (MI.readsRegister(Counter, TRI) || MI.modifiesRegister(Counter, TRI))
      return false;
  }
  if (!Decrement || !Branch)
    return false;

  // Tmp must be dead on leaving the latch.
  for (MachineBasicBlock *Succ : Latch.successors())
    if (Succ->isLiveIn(Tmp))
      return false;

  LLVM_DEBUG(dbgs() << "  folding duplicate counter " << printReg(Tmp, TRI)
                    << " into " << printReg(Counter, TRI) << " in "
                    << printMBBReference(Latch) << "\n");

  // Counter is still live across the backedge, so none of its uses kill it.
  LoopEnd.getOperand(0).setReg(Counter);
  LoopEnd.getOperand(1).setReg(Counter);
  LoopEnd.getOperand(1).setIsKill(false);

  for (MachineOperand &MO : Branch->operands())
    if (MO.isReg() && MO.getReg() == Tmp) {
      MO.setReg(Counter);
      MO.setIsKill(false);
    }

  Copy->eraseFromParent();
  Decrement->eraseFromParent();
  return true;
}

bool RISCVHardwareLoops::tryConvertCandidate(CVHWLoopCandidate &Candidate,
                                             unsigned LoopID, bool IsOuter) {
  assert(LoopID < 2 && "CV32E40P provides two hardware-loop register sets");

  LLVM_DEBUG(dbgs() << "Considering " << printMBBReference(*Candidate.Header)
                    << " for hardware loop " << LoopID << '\n');

  std::optional<CVHWLoopCount> Count = getLoopCount(*Candidate.Setup);
  if (!Count) {
    LLVM_DEBUG(dbgs() << "  unusable trip count\n");
    return false;
  }

  findDeadSoftwareCounter(Candidate, *Count, *TRI);

  std::optional<CVHWLoopBody> Body = measureBody(Candidate);
  if (!Body) {
    LLVM_DEBUG(dbgs() << "  body is not a legal hardware loop body\n");
    return false;
  }

  if (Body->RelaxableInstructions)
    LLVM_DEBUG(dbgs() << "  body has " << Body->RelaxableInstructions
                      << " instructions the linker could delete\n");

  unsigned PadCount = 0;
  const unsigned MinInstructions =
      Body->NumInstructions > Body->RelaxableInstructions
          ? Body->NumInstructions - Body->RelaxableInstructions
          : 0;
  if (MinInstructions < CVMinBodyInstructions) {
    // A body below the minimum can be padded with nops, at the cost of one
    // wasted cycle per iteration for each.
    if (!PadShortBodies) {
      LLVM_DEBUG(dbgs() << "  body has fewer than " << CVMinBodyInstructions
                        << " instructions\n");
      return false;
    }

    PadCount = CVMinBodyInstructions - MinInstructions;
    Body->NumInstructions += PadCount;
    Body->InstructionsAfterInnerEnd += PadCount;
    Body->SizeInBytes += PadCount * UncompressedInstrSize;
  }

  // An outer loop can only be converted once its inner loop has been, since the
  // separation is measured against the inner loop's end marker.
  if (IsOuter) {
    const uint64_t MinAfterInnerEnd =
        Body->InstructionsAfterInnerEnd > Body->RelaxableAfterInnerEnd
            ? Body->InstructionsAfterInnerEnd - Body->RelaxableAfterInnerEnd
            : 0;

    if (!Body->SawInnerEnd || MinAfterInnerEnd < CVMinNestedEndSeparation) {
      LLVM_DEBUG(dbgs() << "  insufficient separation from the inner loop end\n");
      return false;
    }
  }

  if (isCountClobberedAtSetupPoint(Candidate, *Count, *TRI)) {
    std::optional<Register> Copy = findCountCopyAtSetupPoint(Candidate, *Count, *TRI);
    if (!Copy) {
      LLVM_DEBUG(dbgs() << "  trip count is clobbered before the setup point\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "  trip count is clobbered; reading its copy "
                      << printReg(*Copy, TRI) << " instead\n");
    Count->Reg = *Copy;
  }

  // An exit block already pinned by another hardware loop (typically a shared
  // function epilogue) cannot be moved.
  const bool ExitFollowsLatch =
      Candidate.Latch->getNextNode() == Candidate.Exit;
  const bool UseTrampoline =
      !ExitFollowsLatch && PinnedBlocks.contains(Candidate.Exit);

  if (UseTrampoline ? !canInsertExitTrampoline(Candidate)
                    : !canPlaceExitAfterLatch(Candidate)) {
    LLVM_DEBUG(dbgs() << "  exit block cannot follow the latch\n");
    return false;
  }

  // The preheader successor can only change when the exit block currently
  // follows it.
  const bool ExitWillMove = !ExitFollowsLatch && !UseTrampoline;
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

  if (UseTrampoline) {
    LLVM_DEBUG(dbgs() << "  exit " << printMBBReference(*Candidate.Exit)
                      << " is pinned by another hardware loop; using a jump\n");
    Candidate.Exit = insertExitTrampoline(Candidate, *TII);
  } else {
    placeExitAfterLatch(Candidate, *TII);
  }
  assert(Candidate.Preheader->getNextNode() == Candidate.Header &&
         "adjacency prediction disagreed with the resulting layout");
  assert(Candidate.Latch->getNextNode() == Candidate.Exit &&
         "the exit block must directly follow the latch");

  // Pad the loop body (if needed).
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

  for (MachineInstr *MI : Candidate.DeadCounter)
    MI->eraseFromParent();

  Candidate.Latch->removeSuccessor(Candidate.Header,
                                   /*NormalizeSuccProbs=*/ true);

  PinnedBlocks.insert(Candidate.Preheader);
  PinnedBlocks.insert(Candidate.Exit);

  LLVM_DEBUG(dbgs() << "  converted using "
                    << (UseCombinedSetup ? "cv.setup" : "cv.starti/cv.endi")
                    << ", body " << Body->SizeInBytes << " bytes\n");
  ++NumHardwareLoops;
  return true;
}

bool RISCVHardwareLoops::processLoopTree(MachineLoop &ML) {
  LLVM_DEBUG(dbgs() << "  processLoopTree: subloops="
                    << ML.getSubLoops().size() << "\n");
  ArrayRef<MachineLoop *> SubLoops = ML.getSubLoops();

  // An innermost loop uses hardware-loop register set 0.
  if (SubLoops.empty()) {
    std::optional<CVHWLoopCandidate> Candidate = analyzeLoop(ML);
    LLVM_DEBUG(dbgs() << "  analyzeLoop returned "
                      << (Candidate ? "a candidate" : "nullopt") << "\n");
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

  // Sibling loops can share their exit block, which can directly follow only
  // one of their latches; the others reach it through a jump on every exit.
  // The first loop converted keeps the fall-through, so convert the most
  // frequently entered loop first.
  bool Changed = false;
  for (MachineLoop *SubLoop : sortByEntryFrequency(SubLoops))
    Changed |= processLoopTree(*SubLoop);

  return Changed;
}

SmallVector<MachineLoop *, 4> RISCVHardwareLoops::sortByEntryFrequency(
    ArrayRef<MachineLoop *> Loops) const {
  SmallVector<MachineLoop *, 4> Sorted(Loops.begin(), Loops.end());
  auto EntryFreq = [&](const MachineLoop *ML) {
    const MachineBasicBlock *Preheader = ML->getLoopPreheader();
    return Preheader ? MBFI->getBlockFreq(Preheader) : BlockFrequency(0);
  };
  // Without a profile, the guards that choose between sibling loops are often
  // even. Then prefer the larger loop (more blocks, so usually a nest): the
  // other one is typically the special case, e.g. the version of a loop that
  // LoopUnswitch created for an empty inner loop. Otherwise keep the order.
  llvm::stable_sort(Sorted, [&](const MachineLoop *A, const MachineLoop *B) {
    BlockFrequency FA = EntryFreq(A), FB = EntryFreq(B);
    if (FA != FB)
      return FA > FB;
    return A->getNumBlocks() > B->getNumBlocks();
  });
  LLVM_DEBUG({
    for (const MachineLoop *ML : Sorted)
      dbgs() << "  sibling " << printMBBReference(*ML->getHeader())
             << " entry frequency " << EntryFreq(ML).getFrequency() << "\n";
  });
  return Sorted;
}

bool RISCVHardwareLoops::runOnMachineFunction(MachineFunction &MF) {
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  LLVM_DEBUG(dbgs() << "RISCVHardwareLoops pass: " << MF.getName() << "\n");

  if (!ST.hasVendorXCVhwlp())
    return false;

  if (DisableHardwareLoops || skipFunction(MF.getFunction()))
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  PinnedBlocks.clear();

  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  // Snapshot the top-level loops: MachineLoopInfo is required but deliberately
  // not preserved, and becomes stale as soon as a backedge is removed.
  SmallVector<MachineLoop *, 8> TopLevelLoops(MLI.begin(), MLI.end());
  LLVM_DEBUG(dbgs() << "  top-level loops: " << TopLevelLoops.size() << "\n");

  bool Changed = false;
  for (MachineLoop *TopLevelLoop : TopLevelLoops)
    Changed |= processLoopTree(*TopLevelLoop);

  // Loops left in software keep their loop-end pseudo.
  SmallVector<MachineInstr *, 8> LoopEnds;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == RISCV::PseudoCVHWLoopEnd)
        LoopEnds.push_back(&MI);
  for (MachineInstr *LoopEnd : LoopEnds)
    Changed |= foldDuplicateCounter(*LoopEnd);

  if (Changed)
    MF.RenumberBlocks();

  return Changed;
}
