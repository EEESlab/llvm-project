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
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVealu, IsRV32] in {
} // Predicates = [HasVendorXCVealu, IsRV32]
