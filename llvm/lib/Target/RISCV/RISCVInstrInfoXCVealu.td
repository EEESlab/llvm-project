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
// These instructions are only available for riscv32.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// rd(pair) = rs1(pair) op rs2(pair): the R-type format of CVInstAluRR, with
// every operand a register pair.
class CVInstEaluRR<bits<7> funct7, bits<3> funct3, string opcodestr,
                   bit Commutable = 0>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPRPairRV32:$rd),
              (ins GPRPairRV32:$rs1, GPRPairRV32:$rs2), opcodestr,
              "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVealu, IsRV32] in {
  def CV_ADD_D : CVInstEaluRR<0b0010000, 0b000, "cv.add.d", Commutable=1>,
                 Sched<[]>;
  def CV_SUB_D : CVInstEaluRR<0b0110000, 0b000, "cv.sub.d">, Sched<[]>;
} // Predicates = [HasVendorXCVealu, IsRV32]

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// A 64-bit add/sub is legalised to the two-result RISCVISD::ADDD/SUBD node
// (RISCVISelLowering.cpp, ReplaceNodeResults) and selected to CV_ADD_D/CV_SUB_D
// in RISCVISelDAGToDAG.cpp, where the register pair is built with REG_SEQUENCE:
// an i64 is not a legal type on RV32, so no tablegen pattern can match it.
