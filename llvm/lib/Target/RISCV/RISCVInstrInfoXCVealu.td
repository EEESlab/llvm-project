//===-- RISCVInstrInfoXCVealu.td - CORE-V 64-bit ALU -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the CORE-V XCVealu extension: 64-bit integer ALU
// operations on pairs of general purpose registers (GAP "ealu64"). A 64-bit
// value lives in an even/odd register pair, the low word in the even register
// (GPRPair, as the Zacas and P extensions use on RV32).
//
// The encodings are those of the GAP toolchain (riscv32-unknown-elf-as,
// -march=rv32imcxgap9): R-type in the OP opcode space, I-type in OP-IMM and,
// for the comparisons and logical operations with an immediate, in the
// OP-IMM-32 space, which RV32 leaves free. The mnemonics are the GAP ones
// with the cv. prefix (GAP add.d -> cv.add.d).
//
// Two encodings collide with standard extensions: cv.xor.d is Zba sh2add and
// cv.sra.d is Zbb ror. The instructions decode in the XCV namespace, tried
// before the standard tables, and XCVealu is declared incompatible with Zba
// and Zbb in RISCVISAInfo.cpp.
//
// These instructions are only available for riscv32.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SDNode definitions
//===----------------------------------------------------------------------===//

// Two-result nodes on split i64 values (lo, hi) = op(lo1, hi1, lo2, hi2), the
// profile of the P extension ADDD/SUBD nodes (RISCVInstrInfoP.td), for the
// operations P has no node for.
def riscv_andd : RVSDNode<"ANDD", SDT_RISCVIntBinOpD,
                          [SDNPCommutative, SDNPAssociative]>;
def riscv_ord  : RVSDNode<"ORD", SDT_RISCVIntBinOpD,
                          [SDNPCommutative, SDNPAssociative]>;
def riscv_xord : RVSDNode<"XORD", SDT_RISCVIntBinOpD,
                          [SDNPCommutative, SDNPAssociative]>;
def riscv_smind : RVSDNode<"SMIND", SDT_RISCVIntBinOpD,
                           [SDNPCommutative, SDNPAssociative]>;
def riscv_umind : RVSDNode<"UMIND", SDT_RISCVIntBinOpD,
                           [SDNPCommutative, SDNPAssociative]>;
def riscv_smaxd : RVSDNode<"SMAXD", SDT_RISCVIntBinOpD,
                           [SDNPCommutative, SDNPAssociative]>;
def riscv_umaxd : RVSDNode<"UMAXD", SDT_RISCVIntBinOpD,
                           [SDNPCommutative, SDNPAssociative]>;
// Shifts: the amount is a register pair too (GAP sll.d rs2 is a pair), so the
// node keeps the four-input profile with (amount, 0) as the second pair.
def riscv_slld : RVSDNode<"SLLD", SDT_RISCVIntBinOpD>;
def riscv_srld : RVSDNode<"SRLD", SDT_RISCVIntBinOpD>;
def riscv_srad : RVSDNode<"SRAD", SDT_RISCVIntBinOpD>;

// (lo, hi) = op(lo1, hi1): one pair in, one pair out, the generic two-result
// two-input integer profile (the one of the P extension WADDU/WMULSU nodes).
def riscv_absd : RVSDNode<"ABSD", SDTIntBinHiLoOp>;

// rd = op(lo1, hi1): one 32-bit result from one pair, the generic one-result
// two-input integer profile.
def riscv_cntd : RVSDNode<"CNTD", SDTIntBinOp>;

// (lo, hi) = ext(rs1): one pair from one 32-bit register, the halfword (or
// byte) in it extended to 64 bits.
def SDT_RISCVIntExtOpD : SDTypeProfile<2, 1, [SDTCisVT<0, i32>,
                                              SDTCisSameAs<0, 1>,
                                              SDTCisSameAs<0, 2>]>;
def riscv_exthsd : RVSDNode<"EXTHSD", SDT_RISCVIntExtOpD>;
def riscv_exthzd : RVSDNode<"EXTHZD", SDT_RISCVIntExtOpD>;
def riscv_extbsd : RVSDNode<"EXTBSD", SDT_RISCVIntExtOpD>;
def riscv_extbzd : RVSDNode<"EXTBZD", SDT_RISCVIntExtOpD>;

// rd = (lo1, hi1) cmp (lo2, hi2): one 32-bit result from two pairs.
def SDT_RISCVIntCmpOpD : SDTypeProfile<1, 4, [SDTCisVT<0, i32>,
                                              SDTCisVT<1, i32>,
                                              SDTCisSameAs<1, 2>,
                                              SDTCisSameAs<1, 3>,
                                              SDTCisSameAs<1, 4>]>;
def riscv_sltd : RVSDNode<"SLTD", SDT_RISCVIntCmpOpD>;
def riscv_sltud : RVSDNode<"SLTUD", SDT_RISCVIntCmpOpD>;
def riscv_seqd : RVSDNode<"SEQD", SDT_RISCVIntCmpOpD>;
def riscv_sned : RVSDNode<"SNED", SDT_RISCVIntCmpOpD>;
def riscv_sletd : RVSDNode<"SLETD", SDT_RISCVIntCmpOpD>;
def riscv_sletud : RVSDNode<"SLETUD", SDT_RISCVIntCmpOpD>;

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// rd = rs1(pair) op rs2(pair): the R-type format of CVInstAluRR, with the
// sources register pairs. rd is a pair too, or a GPR for the comparisons.
class CVInstEaluRR<bits<7> funct7, bits<3> funct3, string opcodestr,
                   bit Commutable = 0, DAGOperand Rd = GPRPairRV32>
    : RVInstR<funct7, funct3, OPC_OP, (outs Rd:$rd),
              (ins GPRPairRV32:$rs1, GPRPairRV32:$rs2), opcodestr,
              "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}

// rd = op(rs1): the R-type format of CVInstAluR, rs2 unused. rd is a pair,
// or a GPR for the population count; rs1 is a pair, or a GPR for the
// extensions.
class CVInstEaluR<bits<7> funct7, bits<3> funct3, string opcodestr,
                  DAGOperand Rd = GPRPairRV32, DAGOperand Rs1 = GPRPairRV32>
    : RVInstR<funct7, funct3, OPC_OP, (outs Rd:$rd), (ins Rs1:$rs1),
              opcodestr, "$rd, $rs1"> {
  let rs2 = 0b00000;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVealu, IsRV32], DecoderNamespace = "XCV" in {
  def CV_ADD_D : CVInstEaluRR<0b0010000, 0b000, "cv.add.d", Commutable=1>,
                 Sched<[]>;
  def CV_SUB_D : CVInstEaluRR<0b0110000, 0b000, "cv.sub.d">, Sched<[]>;
  def CV_AND_D : CVInstEaluRR<0b0110000, 0b111, "cv.and.d", Commutable=1>,
                 Sched<[]>;
  def CV_OR_D  : CVInstEaluRR<0b0110000, 0b110, "cv.or.d", Commutable=1>,
                 Sched<[]>;
  def CV_XOR_D : CVInstEaluRR<0b0010000, 0b100, "cv.xor.d", Commutable=1>,
                 Sched<[]>;
  def CV_MIN_D : CVInstEaluRR<0b0010010, 0b100, "cv.min.d", Commutable=1>,
                 Sched<[]>;
  def CV_MINU_D : CVInstEaluRR<0b0010010, 0b101, "cv.minu.d", Commutable=1>,
                  Sched<[]>;
  def CV_MAX_D : CVInstEaluRR<0b0010010, 0b110, "cv.max.d", Commutable=1>,
                 Sched<[]>;
  def CV_MAXU_D : CVInstEaluRR<0b0010010, 0b111, "cv.maxu.d", Commutable=1>,
                  Sched<[]>;
  def CV_SLL_D : CVInstEaluRR<0b0010000, 0b001, "cv.sll.d">, Sched<[]>;
  def CV_SRL_D : CVInstEaluRR<0b0010000, 0b101, "cv.srl.d">, Sched<[]>;
  def CV_SRA_D : CVInstEaluRR<0b0110000, 0b101, "cv.sra.d">, Sched<[]>;
  def CV_ABS_D : CVInstEaluR<0b0010010, 0b000, "cv.abs.d">, Sched<[]>;
  def CV_SLT_D : CVInstEaluRR<0b0011000, 0b010, "cv.slt.d", Rd=GPR>, Sched<[]>;
  def CV_SLTU_D : CVInstEaluRR<0b0011000, 0b011, "cv.sltu.d", Rd=GPR>,
                  Sched<[]>;
  def CV_SEQ_D : CVInstEaluRR<0b0011011, 0b010, "cv.seq.d", Rd=GPR>, Sched<[]>;
  def CV_SNE_D : CVInstEaluRR<0b0011011, 0b011, "cv.sne.d", Rd=GPR>, Sched<[]>;
  def CV_SLET_D : CVInstEaluRR<0b0011010, 0b010, "cv.slet.d", Rd=GPR>,
                  Sched<[]>;
  def CV_SLETU_D : CVInstEaluRR<0b0011010, 0b011, "cv.sletu.d", Rd=GPR>,
                   Sched<[]>;
  def CV_CNT_D : CVInstEaluR<0b0011010, 0b001, "cv.cnt.d", Rd=GPR>, Sched<[]>;
  def CV_EXTHS_D : CVInstEaluR<0b0110010, 0b000, "cv.exths.d", Rs1=GPR>,
                   Sched<[]>;
  def CV_EXTHZ_D : CVInstEaluR<0b0110010, 0b001, "cv.exthz.d", Rs1=GPR>,
                   Sched<[]>;
  def CV_EXTBS_D : CVInstEaluR<0b0110010, 0b010, "cv.extbs.d", Rs1=GPR>,
                   Sched<[]>;
  def CV_EXTBZ_D : CVInstEaluR<0b0110010, 0b011, "cv.extbz.d", Rs1=GPR>,
                   Sched<[]>;
  def CV_EXTWS_D : CVInstEaluR<0b0110010, 0b100, "cv.extws.d", Rs1=GPR>,
                   Sched<[]>;
} // Predicates = [HasVendorXCVealu, IsRV32], DecoderNamespace = "XCV"

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// A 64-bit add/sub/and/or/xor/smin/umin/smax/umax/shl/srl/sra/abs is
// legalised to the two-result RISCVISD::ADDD/SUBD/ANDD/ORD/XORD/SMIND/UMIND/
// SMAXD/UMAXD/SLLD/SRLD/SRAD/ABSD node, a 64-bit population count to the
// one-result RISCVISD::CNTD node, a 64-bit zero extension of a halfword/byte
// to the two-result RISCVISD::EXTHZD/EXTBZD node (RISCVISelLowering.cpp,
// ReplaceNodeResults), a 64-bit sign extension of a halfword/byte to the
// two-result RISCVISD::EXTHSD/EXTBSD node (performSRACombine, after type
// legalisation), a 64-bit compare to the one-result RISCVISD::SLTD/SLTUD/
// SEQD/SNED/SLETD/SLETUD node (LowerOperation); they are selected to
// CV_ADD_D/CV_SUB_D/CV_AND_D/CV_OR_D/CV_XOR_D/CV_MIN_D/CV_MINU_D/CV_MAX_D/
// CV_MAXU_D/CV_SLL_D/CV_SRL_D/CV_SRA_D/CV_ABS_D/CV_CNT_D/CV_EXTHS_D/
// CV_EXTHZ_D/CV_EXTBS_D/CV_EXTBZ_D/CV_SLT_D/CV_SLTU_D/CV_SEQ_D/CV_SNE_D/
// CV_SLET_D/CV_SLETU_D in
// RISCVISelDAGToDAG.cpp, where the register pair is built with REG_SEQUENCE:
// an i64 is not a legal type on RV32, so no tablegen pattern can match it.
// CV_EXTWS_D has no selection: the default expansion of a 64-bit sign
// extension of a word is already one instruction (srai of the high word).
