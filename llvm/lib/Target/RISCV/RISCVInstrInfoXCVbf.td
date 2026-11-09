//===-- RISCVInstrInfoXCVbf.td - CORE-V bfloat16 -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the CORE-V bfloat16 extensions, with the values in the
// integer registers (Zfinx style):
//
//  - XCVbf: scalar bfloat16 in the low 16 bits of a GPR. Each instruction is
//    the Zfh binary16 instruction with funct3 set to a value that Zfh leaves
//    unused: 0b101 for the instructions that round (a reserved rounding mode
//    for Zfh) and the Zfh funct3 with bit 2 set for the others (sign
//    injection, min/max, compares, classify). The rounding instructions have
//    no rounding-mode operand. cv.fcvt.h.bf16 is the exception: its funct3 is
//    a real rounding mode.
//
//  - XCVbfv: two bfloat16 lanes packed in one GPR (lane 0 in bits 15:0). The
//    encodings are those of XCVhv with the format field set to 0b01 instead
//    of 0b10 (RISCVInstrInfoXCVhv.td).
//
// cv.fcvt.s.bf16 has the encoding of Zfbfmin's fcvt.s.bf16. The two cannot be
// enabled together: Zfbfmin requires F, XCVbf requires Zfinx.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions
//===----------------------------------------------------------------------===//

// A bfloat16 operand in a GPR. It parses and prints exactly like Zhinx's
// FPR16INX; only the value type seen by the selection patterns differs.
def FPR16BFINX : RegisterOperand<GPRBF16> {
  let ParserMatchClass = GPRAsFPR16;
}

// A v2bf16 operand: a GPR typed v2bf16.
def GPRV2BF16Op : RegisterOperand<GPRV2BF16> {
  let DecoderMethod = "DecodeGPRRegisterClass";
}

//===----------------------------------------------------------------------===//
// XCVbf: scalar instruction class templates
//===----------------------------------------------------------------------===//

defvar XCVBFRound = 0b101;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in {
class CVBF_rr<bits<7> funct7, bits<3> funct3, string opcodestr,
              bit Commutable = 0, DAGOperand rdty = FPR16BFINX>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd),
              (ins FPR16BFINX:$rs1, FPR16BFINX:$rs2),
              "cv." # opcodestr # ".bf16", "$rd, $rs1, $rs2"> {
  let isCommutable = Commutable;
}

class CVBF_r<bits<7> funct7, bits<5> rs2val, bits<3> funct3,
             DAGOperand rdty, DAGOperand rs1ty, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd), (ins rs1ty:$rs1),
              opcodestr, "$rd, $rs1"> {
  let rs2 = rs2val;
}

let isCommutable = 1 in
class CVBF_fma<RISCVOpcode opcode, string opcodestr>
    : RVInstR4<0b10, XCVBFRound, opcode, (outs FPR16BFINX:$rd),
               (ins FPR16BFINX:$rs1, FPR16BFINX:$rs2, FPR16BFINX:$rs3),
               "cv." # opcodestr # ".bf16", "$rd, $rs1, $rs2, $rs3">,
      Sched<[WriteFMA16, ReadFMA16, ReadFMA16, ReadFMA16Addend]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1

//===----------------------------------------------------------------------===//
// XCVbf: scalar instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVbf], DecoderNamespace = "XCV" in {
def CV_FMADD_BF16  : CVBF_fma<OPC_MADD,  "fmadd">;
def CV_FMSUB_BF16  : CVBF_fma<OPC_MSUB,  "fmsub">;
def CV_FNMSUB_BF16 : CVBF_fma<OPC_NMSUB, "fnmsub">;
def CV_FNMADD_BF16 : CVBF_fma<OPC_NMADD, "fnmadd">;

def CV_FADD_BF16 : CVBF_rr<0b0000010, XCVBFRound, "fadd", /*Commutable*/1>,
                   Sched<[WriteFAdd16, ReadFAdd16, ReadFAdd16]>;
def CV_FSUB_BF16 : CVBF_rr<0b0000110, XCVBFRound, "fsub">,
                   Sched<[WriteFAdd16, ReadFAdd16, ReadFAdd16]>;
def CV_FMUL_BF16 : CVBF_rr<0b0001010, XCVBFRound, "fmul", /*Commutable*/1>,
                   Sched<[WriteFMul16, ReadFMul16, ReadFMul16]>;
def CV_FDIV_BF16 : CVBF_rr<0b0001110, XCVBFRound, "fdiv">,
                   Sched<[WriteFDiv16, ReadFDiv16, ReadFDiv16]>;
def CV_FSQRT_BF16 : CVBF_r<0b0101110, 0b00000, XCVBFRound, FPR16BFINX,
                           FPR16BFINX, "cv.fsqrt.bf16">,
                    Sched<[WriteFSqrt16, ReadFSqrt16]>;

let mayRaiseFPException = 0 in {
def CV_FSGNJ_BF16  : CVBF_rr<0b0010010, 0b100, "fsgnj">,
                     Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
def CV_FSGNJN_BF16 : CVBF_rr<0b0010010, 0b101, "fsgnjn">,
                     Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
def CV_FSGNJX_BF16 : CVBF_rr<0b0010010, 0b110, "fsgnjx">,
                     Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
}
def CV_FMIN_BF16 : CVBF_rr<0b0010110, 0b100, "fmin", /*Commutable*/1>,
                   Sched<[WriteFMinMax16, ReadFMinMax16, ReadFMinMax16]>;
def CV_FMAX_BF16 : CVBF_rr<0b0010110, 0b101, "fmax", /*Commutable*/1>,
                   Sched<[WriteFMinMax16, ReadFMinMax16, ReadFMinMax16]>;

let IsSignExtendingOpW = 1 in {
def CV_FEQ_BF16 : CVBF_rr<0b1010010, 0b110, "feq", /*Commutable*/1, GPR>,
                  Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
def CV_FLT_BF16 : CVBF_rr<0b1010010, 0b101, "flt", /*Commutable*/0, GPR>,
                  Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
def CV_FLE_BF16 : CVBF_rr<0b1010010, 0b100, "fle", /*Commutable*/0, GPR>,
                  Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
}

let mayRaiseFPException = 0 in
def CV_FCLASS_BF16 : CVBF_r<0b1110010, 0b00000, XCVBFRound, GPR, FPR16BFINX,
                            "cv.fclass.bf16">,
                     Sched<[WriteFClass16, ReadFClass16]>;

// bfloat16 <-> 32-bit integer, rounding as the arithmetic does.
def CV_FCVT_W_BF16  : CVBF_r<0b1100010, 0b00000, XCVBFRound, GPR, FPR16BFINX,
                             "cv.fcvt.w.bf16">,
                      Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def CV_FCVT_WU_BF16 : CVBF_r<0b1100010, 0b00001, XCVBFRound, GPR, FPR16BFINX,
                             "cv.fcvt.wu.bf16">,
                      Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def CV_FCVT_BF16_W  : CVBF_r<0b1101010, 0b00000, XCVBFRound, FPR16BFINX, GPR,
                             "cv.fcvt.bf16.w">,
                      Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
def CV_FCVT_BF16_WU : CVBF_r<0b1101010, 0b00001, XCVBFRound, FPR16BFINX, GPR,
                             "cv.fcvt.bf16.wu">,
                      Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;

// bfloat16 <-> float. cv.fcvt.s.bf16 is exact (funct3 000);
// cv.fcvt.bf16.s rounds.
def CV_FCVT_S_BF16 : CVBF_r<0b0100000, 0b00110, 0b000, FPR32INX, FPR16BFINX,
                            "cv.fcvt.s.bf16">,
                     Sched<[WriteFCvtF16ToF32, ReadFCvtF16ToF32]>;
def CV_FCVT_BF16_S : CVBF_r<0b0100010, 0b00000, XCVBFRound, FPR16BFINX,
                            FPR32INX, "cv.fcvt.bf16.s">,
                     Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
} // Predicates = [HasVendorXCVbf], DecoderNamespace = "XCV"

// bfloat16 <-> binary16. cv.fcvt.h.bf16 takes a rounding mode (dynamic by
// default); cv.fcvt.bf16.h rounds as the other bfloat16 instructions.
let Predicates = [HasVendorXCVbf, HasStdExtZhinx], DecoderNamespace = "XCV" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
def CV_FCVT_H_BF16 : RVInstRFrm<0b0100010, OPC_OP_FP, (outs FPR16INX:$rd),
                                (ins FPR16BFINX:$rs1, frmarg:$frm),
                                "cv.fcvt.h.bf16", "$rd, $rs1$frm">,
                     Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]> {
  let rs2 = 0b00110;
}
def CV_FCVT_BF16_H : CVBF_r<0b0100010, 0b00010, XCVBFRound, FPR16BFINX,
                            FPR16INX, "cv.fcvt.bf16.h">,
                     Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
}

// Assembler-only aliases.
let Predicates = [HasVendorXCVbf] in {
def : InstAlias<"cv.fabs.bf16 $rd, $rs",
                (CV_FSGNJX_BF16 FPR16BFINX:$rd, FPR16BFINX:$rs, FPR16BFINX:$rs), 0>;
def : InstAlias<"cv.fneg.bf16 $rd, $rs",
                (CV_FSGNJN_BF16 FPR16BFINX:$rd, FPR16BFINX:$rs, FPR16BFINX:$rs), 0>;
def : InstAlias<"cv.fgt.bf16 $rd, $rs, $rt",
                (CV_FLT_BF16 GPR:$rd, FPR16BFINX:$rt, FPR16BFINX:$rs), 0>;
def : InstAlias<"cv.fge.bf16 $rd, $rs, $rt",
                (CV_FLE_BF16 GPR:$rd, FPR16BFINX:$rt, FPR16BFINX:$rs), 0>;
}

//===----------------------------------------------------------------------===//
// XCVbfv: packed instruction class templates
//===----------------------------------------------------------------------===//

defvar XCVBFVFmt = 0b01;

// As XCVhv: the packed ops that round use the dynamic rounding mode (frm).
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    Uses = [FRM] in {
class CVBFV_rr<bits<5> op, bit r, dag outs, dag ins, string opcodestr>
    : RVInstR<{0b10, op}, {r, XCVBFVFmt}, OPC_OP, outs, ins, opcodestr,
              "$rd, $rs1, $rs2">;

class CVBFV_r<bits<5> op, bits<5> rs2val, bit r, string opcodestr,
              DAGOperand rdty = GPRV2BF16Op, DAGOperand rs1ty = GPRV2BF16Op>
    : RVInstR<{0b10, op}, {r, XCVBFVFmt}, OPC_OP, (outs rdty:$rd),
              (ins rs1ty:$rs1), opcodestr, "$rd, $rs1"> {
  let rs2 = rs2val;
}

let Constraints = "$rd = $rd_wb" in
class CVBFV_rrr<bits<5> op, bit r, DAGOperand rs2ty, string opcodestr>
    : RVInstR<{0b10, op}, {r, XCVBFVFmt}, OPC_OP, (outs GPRV2BF16Op:$rd_wb),
              (ins GPRV2BF16Op:$rd, GPRV2BF16Op:$rs1, rs2ty:$rs2), opcodestr,
              "$rd, $rs1, $rs2">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
  // Uses = [FRM]

multiclass CVBFV_rr_m<bits<5> op, string mnemonic> {
  def _BF16 : CVBFV_rr<op, 0, (outs GPRV2BF16Op:$rd),
                       (ins GPRV2BF16Op:$rs1, GPRV2BF16Op:$rs2),
                       "cv." # mnemonic # ".bf16">;
  def _R_BF16 : CVBFV_rr<op, 1, (outs GPRV2BF16Op:$rd),
                         (ins GPRV2BF16Op:$rs1, FPR16BFINX:$rs2),
                         "cv." # mnemonic # ".r.bf16">;
}

multiclass CVBFV_rrr_m<bits<5> op, string mnemonic> {
  def _BF16   : CVBFV_rrr<op, 0, GPRV2BF16Op, "cv." # mnemonic # ".bf16">;
  def _R_BF16 : CVBFV_rrr<op, 1, FPR16BFINX,
                          "cv." # mnemonic # ".r.bf16">;
}

//===----------------------------------------------------------------------===//
// XCVbfv: packed instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVbfv, IsRV32], DecoderNamespace = "XCV" in {
defm CV_VFADD : CVBFV_rr_m<0b00001, "vfadd">,
                Sched<[WriteFAdd16, ReadFAdd16, ReadFAdd16]>;
defm CV_VFSUB : CVBFV_rr_m<0b00010, "vfsub">,
                Sched<[WriteFAdd16, ReadFAdd16, ReadFAdd16]>;
defm CV_VFMUL : CVBFV_rr_m<0b00011, "vfmul">,
                Sched<[WriteFMul16, ReadFMul16, ReadFMul16]>;
defm CV_VFDIV : CVBFV_rr_m<0b00100, "vfdiv">,
                Sched<[WriteFDiv16, ReadFDiv16, ReadFDiv16]>;
let Uses = []<Register> in {
defm CV_VFMIN : CVBFV_rr_m<0b00101, "vfmin">,
                Sched<[WriteFMinMax16, ReadFMinMax16, ReadFMinMax16]>;
defm CV_VFMAX : CVBFV_rr_m<0b00110, "vfmax">,
                Sched<[WriteFMinMax16, ReadFMinMax16, ReadFMinMax16]>;
}
def CV_VFSQRT_BF16 : CVBFV_r<0b00111, 0b00000, 0, "cv.vfsqrt.bf16">,
                     Sched<[WriteFSqrt16, ReadFSqrt16]>;

// vfmac: rd = rd + rs1 * rs2, lane-wise.
// vfmre: rd = rs1 * rs2 - rd, lane-wise.
defm CV_VFMAC : CVBFV_rrr_m<0b01000, "vfmac">,
                Sched<[WriteFMA16, ReadFMA16Addend, ReadFMA16, ReadFMA16]>;
defm CV_VFMRE : CVBFV_rrr_m<0b01001, "vfmre">,
                Sched<[WriteFMA16, ReadFMA16Addend, ReadFMA16, ReadFMA16]>;

let mayRaiseFPException = 0, Uses = []<Register> in {
defm CV_VFSGNJ  : CVBFV_rr_m<0b01101, "vfsgnj">,
                  Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
defm CV_VFSGNJN : CVBFV_rr_m<0b01110, "vfsgnjn">,
                  Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
defm CV_VFSGNJX : CVBFV_rr_m<0b01111, "vfsgnjx">,
                  Sched<[WriteFSGNJ16, ReadFSGNJ16, ReadFSGNJ16]>;
def CV_VFCLASS_BF16 : CVBFV_r<0b01100, 0b00001, 0, "cv.vfclass.bf16">,
                      Sched<[WriteFClass16, ReadFClass16]>;
}

// Compares write 0 or 1 to each 16-bit lane of rd.
let Uses = []<Register> in {
defm CV_VFEQ : CVBFV_rr_m<0b10000, "vfeq">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
defm CV_VFNE : CVBFV_rr_m<0b10001, "vfne">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
defm CV_VFLT : CVBFV_rr_m<0b10010, "vflt">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
defm CV_VFGE : CVBFV_rr_m<0b10011, "vfge">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
defm CV_VFLE : CVBFV_rr_m<0b10100, "vfle">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
defm CV_VFGT : CVBFV_rr_m<0b10101, "vfgt">,
               Sched<[WriteFCmp16, ReadFCmp16, ReadFCmp16]>;
}

// Lane-wise bfloat16 <-> 16-bit integer; the r bit selects unsigned.
def CV_VFCVT_X_BF16  : CVBFV_r<0b01100, 0b00010, 0, "cv.vfcvt.x.bf16">,
                       Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def CV_VFCVT_XU_BF16 : CVBFV_r<0b01100, 0b00010, 1, "cv.vfcvt.xu.bf16">,
                       Sched<[WriteFCvtF16ToI32, ReadFCvtF16ToI32]>;
def CV_VFCVT_BF16_X  : CVBFV_r<0b01100, 0b00011, 0, "cv.vfcvt.bf16.x">,
                       Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;
def CV_VFCVT_BF16_XU : CVBFV_r<0b01100, 0b00011, 1, "cv.vfcvt.bf16.xu">,
                       Sched<[WriteFCvtI32ToF16, ReadFCvtI32ToF16]>;

// Rounds two floats to bfloat16: rs1 to lane 0, rs2 to lane 1.
def CV_VFCPKA_BF16_S : CVBFV_rr<0b11000, 0, (outs GPRV2BF16Op:$rd),
                                (ins FPR32INX:$rs1, FPR32INX:$rs2),
                                "cv.vfcpka.bf16.s">,
                       Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16,
                           ReadFCvtF32ToF16]>;
} // Predicates = [HasVendorXCVbfv, IsRV32], DecoderNamespace = "XCV"

// Lane-wise binary16 <-> bfloat16. The format field names the destination
// format: cv.vfcvt.h.bf16 has the XCVhv format (0b10), cv.vfcvt.bf16.h the
// XCVbfv one.
let Predicates = [HasVendorXCVbfv, HasVendorXCVhv, IsRV32],
    DecoderNamespace = "XCV",
    hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    Uses = [FRM] in {
def CV_VFCVT_H_BF16
    : RVInstR<0b1001100, 0b010, OPC_OP, (outs GPRV2F16Op:$rd),
              (ins GPRV2BF16Op:$rs1), "cv.vfcvt.h.bf16", "$rd, $rs1">,
      Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]> {
  let rs2 = 0b00101;
}
def CV_VFCVT_BF16_H : CVBFV_r<0b01100, 0b00110, 0, "cv.vfcvt.bf16.h",
                              GPRV2BF16Op, GPRV2F16Op>,
                      Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
}

// Assembler-only aliases.
let Predicates = [HasVendorXCVbfv, IsRV32] in {
def : InstAlias<"cv.vfabs.bf16 $rd, $rs",
                (CV_VFSGNJX_BF16 GPRV2BF16Op:$rd, GPRV2BF16Op:$rs,
                                 GPRV2BF16Op:$rs), 0>;
def : InstAlias<"cv.vfneg.bf16 $rd, $rs",
                (CV_VFSGNJN_BF16 GPRV2BF16Op:$rd, GPRV2BF16Op:$rs,
                                 GPRV2BF16Op:$rs), 0>;
}

//===----------------------------------------------------------------------===//
// XCVbf: scalar code generation
//===----------------------------------------------------------------------===//

// bf16 is legal in GPRBF16. The ops with an instruction are selected below;
// the others are promoted to f32 (RISCVISelLowering.cpp). Strict FP ops are
// promoted too: the bfloat16 instructions have no rounding-mode operand.
let Predicates = [HasVendorXCVbf] in {
def : Pat<(bf16 (fadd FPR16BFINX:$rs1, FPR16BFINX:$rs2)),
          (CV_FADD_BF16 $rs1, $rs2)>;
def : Pat<(bf16 (fsub FPR16BFINX:$rs1, FPR16BFINX:$rs2)),
          (CV_FSUB_BF16 $rs1, $rs2)>;
def : Pat<(bf16 (fmul FPR16BFINX:$rs1, FPR16BFINX:$rs2)),
          (CV_FMUL_BF16 $rs1, $rs2)>;
def : Pat<(bf16 (fdiv FPR16BFINX:$rs1, FPR16BFINX:$rs2)),
          (CV_FDIV_BF16 $rs1, $rs2)>;
def : Pat<(bf16 (fsqrt FPR16BFINX:$rs1)), (CV_FSQRT_BF16 $rs1)>;

// Sign operations only move the sign bit.
def : Pat<(bf16 (fneg FPR16BFINX:$rs1)), (CV_FSGNJN_BF16 $rs1, $rs1)>;
def : Pat<(bf16 (fabs FPR16BFINX:$rs1)), (CV_FSGNJX_BF16 $rs1, $rs1)>;
def : PatFprFpr<fcopysign, CV_FSGNJ_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<riscv_fsgnjx, CV_FSGNJX_BF16, FPR16BFINX, bf16>;
def : Pat<(bf16 (fcopysign FPR16BFINX:$rs1, (bf16 (fneg FPR16BFINX:$rs2)))),
          (CV_FSGNJN_BF16 $rs1, $rs2)>;
// The sign of a float is bit 31 of its GPR.
def : Pat<(bf16 (fcopysign FPR16BFINX:$rs1, (f32 FPR32INX:$rs2))),
          (CV_FSGNJ_BF16 $rs1, (CV_FCVT_BF16_S $rs2))>;
def : Pat<(f32 (fcopysign FPR32INX:$rs1, (bf16 FPR16BFINX:$rs2))),
          (FSGNJ_S_INX $rs1, (CV_FCVT_S_BF16 $rs2))>;

// Fused multiply-add, as for binary16.
// fmadd: rs1 * rs2 + rs3
def : Pat<(bf16 (fma FPR16BFINX:$rs1, FPR16BFINX:$rs2, FPR16BFINX:$rs3)),
          (CV_FMADD_BF16 $rs1, $rs2, $rs3)>;
// fmsub: rs1 * rs2 - rs3
def : Pat<(bf16 (fma FPR16BFINX:$rs1, FPR16BFINX:$rs2,
                     (fneg FPR16BFINX:$rs3))),
          (CV_FMSUB_BF16 $rs1, $rs2, $rs3)>;
// fnmsub: -rs1 * rs2 + rs3
def : Pat<(bf16 (fma (fneg FPR16BFINX:$rs1), FPR16BFINX:$rs2,
                     FPR16BFINX:$rs3)),
          (CV_FNMSUB_BF16 $rs1, $rs2, $rs3)>;
// fnmadd: -rs1 * rs2 - rs3
def : Pat<(bf16 (fma (fneg FPR16BFINX:$rs1), FPR16BFINX:$rs2,
                     (fneg FPR16BFINX:$rs3))),
          (CV_FNMADD_BF16 $rs1, $rs2, $rs3)>;
// fnmadd: -(rs1 * rs2 + rs3) (the nsz flag on the FMA)
def : Pat<(bf16 (fneg (any_fma_nsz FPR16BFINX:$rs1, FPR16BFINX:$rs2,
                                   FPR16BFINX:$rs3))),
          (CV_FNMADD_BF16 $rs1, $rs2, $rs3)>;

// As fmin.h / fmax.h.
def : PatFprFpr<fminnum, CV_FMIN_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<fmaxnum, CV_FMAX_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<fminimumnum, CV_FMIN_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<fmaximumnum, CV_FMAX_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<riscv_fmin, CV_FMIN_BF16, FPR16BFINX, bf16>;
def : PatFprFpr<riscv_fmax, CV_FMAX_BF16, FPR16BFINX, bf16>;
def : Pat<(bf16 (fcanonicalize FPR16BFINX:$rs1)), (CV_FMIN_BF16 $rs1, $rs1)>;

def : Pat<(riscv_fclass (bf16 FPR16BFINX:$rs1)), (CV_FCLASS_BF16 $rs1)>;

// Compares. The other conditions are expanded onto these.
def : PatSetCC<FPR16BFINX, setcc, SETEQ,  CV_FEQ_BF16, bf16>;
def : PatSetCC<FPR16BFINX, setcc, SETOEQ, CV_FEQ_BF16, bf16>;
def : PatSetCC<FPR16BFINX, setcc, SETLT,  CV_FLT_BF16, bf16>;
def : PatSetCC<FPR16BFINX, setcc, SETOLT, CV_FLT_BF16, bf16>;
def : PatSetCC<FPR16BFINX, setcc, SETLE,  CV_FLE_BF16, bf16>;
def : PatSetCC<FPR16BFINX, setcc, SETOLE, CV_FLE_BF16, bf16>;

defm Select_FPR16BFINX : SelectCC_GPR_rrirr<FPR16BFINX, bf16>;

// Loads and stores: the bf16 is the low half (sub_16) of the GPR that lh
// loads or sh stores.
def : Pat<(bf16 (load (AddrRegImm (XLenVT BasePtr:$rs1), simm12_lo:$imm12))),
          (COPY_TO_REGCLASS (LH_INX BasePtr:$rs1, simm12_lo:$imm12), GPRBF16)>;
def : Pat<(store (bf16 FPR16BFINX:$rs2),
                 (AddrRegImm (XLenVT BasePtr:$rs1), simm12_lo:$imm12)),
          (SH_INX (COPY_TO_REGCLASS FPR16BFINX:$rs2, GPRF16), BasePtr:$rs1,
                  simm12_lo:$imm12)>;

// Moves (no conversion), e.g. bitcasts from and to i16.
def : Pat<(bf16 (riscv_fmv_h_x GPR:$src)), (EXTRACT_SUBREG GPR:$src, sub_16)>;
def : Pat<(riscv_fmv_x_anyexth (bf16 FPR16BFINX:$src)),
          (INSERT_SUBREG (XLenVT (IMPLICIT_DEF)), FPR16BFINX:$src, sub_16)>;

// bfloat16 <-> float. The extension is exact.
def : Pat<(bf16 (any_fpround FPR32INX:$rs1)), (CV_FCVT_BF16_S $rs1)>;
def : Pat<(f32 (any_fpextend (bf16 FPR16BFINX:$rs1))), (CV_FCVT_S_BF16 $rs1)>;

// 32-bit integer -> bfloat16 in one rounding. bfloat16 -> integer is done in
// float (exact extension, then fcvt.w.s with round-towards-zero), which C
// casts need.
def : Pat<(bf16 (sint_to_fp (i32 GPR:$rs1))), (CV_FCVT_BF16_W $rs1)>;
def : Pat<(bf16 (uint_to_fp (i32 GPR:$rs1))), (CV_FCVT_BF16_WU $rs1)>;
} // Predicates = [HasVendorXCVbf]

// bfloat16 <-> binary16. The conversion through float is exact on the way
// in, so one instruction rounds the same way.
let Predicates = [HasVendorXCVbf, HasStdExtZhinx] in {
def : Pat<(f16 (any_fpround (f32 (any_fpextend (bf16 FPR16BFINX:$rs1))))),
          (CV_FCVT_H_BF16 $rs1, FRM_DYN)>;
def : Pat<(bf16 (any_fpround (f32 (any_fpextend (f16 FPR16INX:$rs1))))),
          (CV_FCVT_BF16_H $rs1)>;
}

//===----------------------------------------------------------------------===//
// XCVbfv: packed code generation
//===----------------------------------------------------------------------===//

def cv_bfv_splat
    : PatFrag<(ops node:$s),
              (v2bf16 (build_vector (bf16 node:$s), (bf16 node:$s)))>;

multiclass CVBFVBinPats<SDPatternOperator Op, string Inst> {
  def : Pat<(v2bf16 (Op GPRV2BF16:$rs1, GPRV2BF16:$rs2)),
            (!cast<Instruction>(Inst # "_BF16") GPRV2BF16:$rs1, GPRV2BF16:$rs2)>;
  def : Pat<(v2bf16 (Op GPRV2BF16:$rs1, (cv_bfv_splat FPR16BFINX:$rs2))),
            (!cast<Instruction>(Inst # "_R_BF16") GPRV2BF16:$rs1,
                                                  FPR16BFINX:$rs2)>;
}

let Predicates = [HasVendorXCVbfv, IsRV32] in {
defm : CVBFVBinPats<fadd, "CV_VFADD">;
defm : CVBFVBinPats<fsub, "CV_VFSUB">;
defm : CVBFVBinPats<fmul, "CV_VFMUL">;
defm : CVBFVBinPats<fminnum, "CV_VFMIN">;
defm : CVBFVBinPats<fmaxnum, "CV_VFMAX">;
defm : CVBFVBinPats<fminimumnum, "CV_VFMIN">;
defm : CVBFVBinPats<fmaximumnum, "CV_VFMAX">;
// As XCVhv, fdiv and fsqrt are scalarized onto cv.fdiv.bf16 and
// cv.fsqrt.bf16.

def : Pat<(v2bf16 (fneg GPRV2BF16:$rs1)),
          (CV_VFSGNJN_BF16 GPRV2BF16:$rs1, GPRV2BF16:$rs1)>;
def : Pat<(v2bf16 (fabs GPRV2BF16:$rs1)),
          (CV_VFSGNJX_BF16 GPRV2BF16:$rs1, GPRV2BF16:$rs1)>;
def : Pat<(v2bf16 (fcopysign GPRV2BF16:$rs1, (v2bf16 GPRV2BF16:$rs2))),
          (CV_VFSGNJ_BF16 GPRV2BF16:$rs1, GPRV2BF16:$rs2)>;
def : Pat<(v2bf16 (fcopysign GPRV2BF16:$rs1,
                             (v2bf16 (fneg GPRV2BF16:$rs2)))),
          (CV_VFSGNJN_BF16 GPRV2BF16:$rs1, GPRV2BF16:$rs2)>;

def : Pat<(v2bf16 (fma GPRV2BF16:$rs1, GPRV2BF16:$rs2, GPRV2BF16:$rd)),
          (CV_VFMAC_BF16 GPRV2BF16:$rd, GPRV2BF16:$rs1, GPRV2BF16:$rs2)>;
def : Pat<(v2bf16 (fma GPRV2BF16:$rs1, (cv_bfv_splat FPR16BFINX:$rs2),
                       GPRV2BF16:$rd)),
          (CV_VFMAC_R_BF16 GPRV2BF16:$rd, GPRV2BF16:$rs1, FPR16BFINX:$rs2)>;
def : Pat<(v2bf16 (fma GPRV2BF16:$rs1, GPRV2BF16:$rs2,
                       (v2bf16 (fneg GPRV2BF16:$rd)))),
          (CV_VFMRE_BF16 GPRV2BF16:$rd, GPRV2BF16:$rs1, GPRV2BF16:$rs2)>;
def : Pat<(v2bf16 (fma GPRV2BF16:$rs1, (cv_bfv_splat FPR16BFINX:$rs2),
                       (v2bf16 (fneg GPRV2BF16:$rd)))),
          (CV_VFMRE_R_BF16 GPRV2BF16:$rd, GPRV2BF16:$rs1, FPR16BFINX:$rs2)>;

def : Pat<(v2bf16 (build_vector (bf16 (fpround (f32 FPR32INX:$rs1))),
                                (bf16 (fpround (f32 FPR32INX:$rs2))))),
          (CV_VFCPKA_BF16_S FPR32INX:$rs1, FPR32INX:$rs2)>;

def : Pat<(bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 0)),
          (bf16 (EXTRACT_SUBREG GPRV2BF16:$rs1, sub_16))>;
def : Pat<(bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 1)),
          (bf16 (EXTRACT_SUBREG
                    (i32 (SRLI (i32 (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR)), 16)),
                    sub_16))>;

def : Pat<(v2bf16 (bitconvert (i32 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2BF16)>;
def : Pat<(i32 (bitconvert (v2bf16 GPRV2BF16:$rs1))),
          (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR)>;
def : Pat<(v2bf16 (bitconvert (f32 FPR32INX:$rs1))),
          (COPY_TO_REGCLASS
              (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR32INX:$rs1, sub_32),
              GPRV2BF16)>;
def : Pat<(f32 (bitconvert (v2bf16 GPRV2BF16:$rs1))),
          (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR)), sub_32)>;
} // Predicates = [HasVendorXCVbfv, IsRV32]

let Predicates = [HasVendorXCVbfv, HasVendorXCVhv, IsRV32] in {
def : Pat<(v2bf16 (bitconvert (v2f16 GPRV2F16:$rs1))),
          (COPY_TO_REGCLASS GPRV2F16:$rs1, GPRV2BF16)>;
def : Pat<(v2f16 (bitconvert (v2bf16 GPRV2BF16:$rs1))),
          (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPRV2F16)>;
// Lane-wise binary16 <-> bfloat16. The <2 x float> step is scalarized, so
// the conversion reaches selection lane by lane.
def : Pat<(v2bf16 (build_vector
              (bf16 (fpround (f32 (fpextend
                  (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 0)))))),
              (bf16 (fpround (f32 (fpextend
                  (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 1)))))))),
          (CV_VFCVT_BF16_H GPRV2F16:$rs1)>;
def : Pat<(v2f16 (build_vector
              (f16 (fpround (f32 (fpextend
                  (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 0)))))),
              (f16 (fpround (f32 (fpextend
                  (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 1)))))))),
          (CV_VFCVT_H_BF16 GPRV2BF16:$rs1)>;
}

let Predicates = [HasVendorXCVbfv, HasStdExtP, IsRV32] in {
def : Pat<(v2bf16 (bitconvert (v2i16 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2BF16)>;
def : Pat<(v2i16 (bitconvert (v2bf16 GPRV2BF16:$rs1))),
          (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR)>;
def : Pat<(v2bf16 (bitconvert (v4i8 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2BF16)>;
def : Pat<(v4i8 (bitconvert (v2bf16 GPRV2BF16:$rs1))),
          (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR)>;
}

// Without XCVsimd, a splat that did not fold into a .r form: x | x << 16.
let Predicates = [HasVendorXCVbfv, NoVendorXCVsimd, IsRV32] in
def : Pat<(v2bf16 (build_vector (bf16 FPR16BFINX:$s), (bf16 FPR16BFINX:$s))),
          (COPY_TO_REGCLASS
              (OR (SLLI (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16BFINX:$s,
                                       sub_16), 16),
                  (SRLI (SLLI (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                             FPR16BFINX:$s, sub_16), 16), 16)),
              GPRV2BF16)>;

// Lane moves with XCVsimd, as for v2f16 (see RISCVInstrInfoXCVhv.td).
let Predicates = [HasVendorXCVbfv, HasVendorXCVsimd, IsRV32] in {
def : Pat<(v2bf16 (build_vector (bf16 FPR16BFINX:$lo), (bf16 FPR16BFINX:$hi))),
          (COPY_TO_REGCLASS
              (CV_PACK (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16BFINX:$hi,
                                      sub_16),
                       (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16BFINX:$lo,
                                      sub_16)),
              GPRV2BF16)>;
let AddedComplexity = 10 in {
def : Pat<(v2bf16 (build_vector
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 1)),
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 0)))),
          (COPY_TO_REGCLASS
              (CV_SHUFFLE_SCI_H (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR), 1),
              GPRV2BF16)>;
def : Pat<(v2bf16 (build_vector
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 1)),
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 1)))),
          (COPY_TO_REGCLASS
              (CV_SHUFFLE_SCI_H (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR), 3),
              GPRV2BF16)>;
def : Pat<(v2bf16 (build_vector
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rs1), 0)),
                      (bf16 (vector_extract (v2bf16 GPRV2BF16:$rd), 1)))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2BF16:$rd, GPR),
                           (COPY_TO_REGCLASS GPRV2BF16:$rs1, GPR), 0),
              GPRV2BF16)>;
}
def : Pat<(v2bf16 (build_vector (bf16 FPR16BFINX:$rs1),
                                (bf16 (vector_extract (v2bf16 GPRV2BF16:$rd), 1)))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2BF16:$rd, GPR),
                           (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16BFINX:$rs1,
                                          sub_16), 0),
              GPRV2BF16)>;
def : Pat<(v2bf16 (build_vector (bf16 (vector_extract (v2bf16 GPRV2BF16:$rd), 0)),
                                (bf16 FPR16BFINX:$rs1))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2BF16:$rd, GPR),
                           (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16BFINX:$rs1,
                                          sub_16), 1),
              GPRV2BF16)>;
} // Predicates = [HasVendorXCVbfv, HasVendorXCVsimd, IsRV32]
