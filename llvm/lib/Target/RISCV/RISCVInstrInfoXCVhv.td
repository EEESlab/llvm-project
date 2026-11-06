//===-- RISCVInstrInfoXCVhv.td - CORE-V packed FP16 -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the CORE-V Xcvhv extension: two IEEE binary16 lanes
// packed in one 32-bit GPR (lane 0 in bits 15:0, lane 1 in bits 31:16), with
// the scalar halves in the integer registers as in Zhinx.
//
// Encoding:
//
//   opcode = OP (0b0110011), funct7 = 0b10 ++ op, funct3 = r ++ vfmt,
//   vfmt = 0b10 for binary16, r = 1 selects the .r form in which rs2 is a
//   scalar half (low 16 bits of the GPR) replicated to both lanes.
//
// funct7 = 0b10xxxxx is unused by the standard OP space and by the other
// CORE-V extensions (which live in custom-0..3), so there is no overlap.
//
// Only the binary16 (.h) format is covered here. Alternate formats such as
// bfloat16, and the conversions to and from them, belong to a separate
// extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// A v2f16 operand. It parses, prints and decodes exactly like a GPR; only the
// value type seen by the selection patterns differs.
def GPRV2F16Op : RegisterOperand<GPRV2F16> {
  let DecoderMethod = "DecodeGPRRegisterClass";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

defvar XCVHVFmtH = 0b10;

// The packed ops that round use the dynamic rounding mode (frm). They read
// FRM so they are not moved across a write of frm; the ops that do not round
// (sign injection, min/max, compares, classify) clear it.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    Uses = [FRM] in {
class CVVF_rr<bits<5> op, bit r, dag outs, dag ins, string opcodestr>
    : RVInstR<{0b10, op}, {r, XCVHVFmtH}, OPC_OP, outs, ins, opcodestr,
              "$rd, $rs1, $rs2">, Sched<[]>;

class CVVF_r<bits<5> op, bits<5> rs2val, bit r, string opcodestr>
    : RVInstR<{0b10, op}, {r, XCVHVFmtH}, OPC_OP, (outs GPRV2F16Op:$rd),
              (ins GPRV2F16Op:$rs1), opcodestr, "$rd, $rs1">, Sched<[]> {
  let rs2 = rs2val;
}

// Ops that read rd (vfmac, vfmre): rd is tied, as for cv.mac.
let Constraints = "$rd = $rd_wb" in
class CVVF_rrr<bits<5> op, bit r, DAGOperand rs2ty, string opcodestr>
    : RVInstR<{0b10, op}, {r, XCVHVFmtH}, OPC_OP, (outs GPRV2F16Op:$rd_wb),
              (ins GPRV2F16Op:$rd, GPRV2F16Op:$rs1, rs2ty:$rs2), opcodestr,
              "$rd, $rs1, $rs2">, Sched<[]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
  // Uses = [FRM]

// Plain and .r forms of a two-operand packed op.
multiclass CVVF_rr_m<bits<5> op, string mnemonic,
                     DAGOperand rdty = GPRV2F16Op> {
  def _H : CVVF_rr<op, 0, (outs rdty:$rd),
                   (ins GPRV2F16Op:$rs1, GPRV2F16Op:$rs2),
                   "cv." # mnemonic # ".h">;
  def _R_H : CVVF_rr<op, 1, (outs rdty:$rd),
                     (ins GPRV2F16Op:$rs1, FPR16INX:$rs2),
                     "cv." # mnemonic # ".r.h">;
}

multiclass CVVF_rrr_m<bits<5> op, string mnemonic> {
  def _H   : CVVF_rrr<op, 0, GPRV2F16Op, "cv." # mnemonic # ".h">;
  def _R_H : CVVF_rrr<op, 1, FPR16INX, "cv." # mnemonic # ".r.h">;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXCVhv, IsRV32], DecoderNamespace = "XCV" in {
defm CV_VFADD : CVVF_rr_m<0b00001, "vfadd">;
defm CV_VFSUB : CVVF_rr_m<0b00010, "vfsub">;
defm CV_VFMUL : CVVF_rr_m<0b00011, "vfmul">;
defm CV_VFDIV : CVVF_rr_m<0b00100, "vfdiv">;
let Uses = []<Register> in {
defm CV_VFMIN : CVVF_rr_m<0b00101, "vfmin">;
defm CV_VFMAX : CVVF_rr_m<0b00110, "vfmax">;
}
def CV_VFSQRT_H : CVVF_r<0b00111, 0b00000, 0, "cv.vfsqrt.h">;

// vfmac: rd = rd + rs1 * rs2, lane-wise.
// vfmre: rd = rs1 * rs2 - rd, lane-wise.
defm CV_VFMAC : CVVF_rrr_m<0b01000, "vfmac">;
defm CV_VFMRE : CVVF_rrr_m<0b01001, "vfmre">;

let mayRaiseFPException = 0, Uses = []<Register> in {
defm CV_VFSGNJ  : CVVF_rr_m<0b01101, "vfsgnj">;
defm CV_VFSGNJN : CVVF_rr_m<0b01110, "vfsgnjn">;
defm CV_VFSGNJX : CVVF_rr_m<0b01111, "vfsgnjx">;
def CV_VFCLASS_H : CVVF_r<0b01100, 0b00001, 0, "cv.vfclass.h">;
}

// Compares write 0 or 1 to each 16-bit lane of rd.
let Uses = []<Register> in {
defm CV_VFEQ : CVVF_rr_m<0b10000, "vfeq">;
defm CV_VFNE : CVVF_rr_m<0b10001, "vfne">;
defm CV_VFLT : CVVF_rr_m<0b10010, "vflt">;
defm CV_VFGE : CVVF_rr_m<0b10011, "vfge">;
defm CV_VFLE : CVVF_rr_m<0b10100, "vfle">;
defm CV_VFGT : CVVF_rr_m<0b10101, "vfgt">;
}

// Lane-wise binary16 <-> 16-bit integer; the r bit selects unsigned.
def CV_VFCVT_X_H  : CVVF_r<0b01100, 0b00010, 0, "cv.vfcvt.x.h">;
def CV_VFCVT_XU_H : CVVF_r<0b01100, 0b00010, 1, "cv.vfcvt.xu.h">;
def CV_VFCVT_H_X  : CVVF_r<0b01100, 0b00011, 0, "cv.vfcvt.h.x">;
def CV_VFCVT_H_XU : CVVF_r<0b01100, 0b00011, 1, "cv.vfcvt.h.xu">;

// Rounds two floats to binary16: rs1 to lane 0, rs2 to lane 1. It writes the
// whole of rd, so rd is not tied.
def CV_VFCPKA_H_S : CVVF_rr<0b11000, 0, (outs GPRV2F16Op:$rd),
                            (ins FPR32INX:$rs1, FPR32INX:$rs2),
                            "cv.vfcpka.h.s">;
} // Predicates = [HasVendorXCVhv, IsRV32], DecoderNamespace = "XCV"

// Assembler-only aliases.
let Predicates = [HasVendorXCVhv, IsRV32] in {
def : InstAlias<"cv.vfabs.h $rd, $rs",
                (CV_VFSGNJX_H GPRV2F16Op:$rd, GPRV2F16Op:$rs, GPRV2F16Op:$rs), 0>;
def : InstAlias<"cv.vfneg.h $rd, $rs",
                (CV_VFSGNJN_H GPRV2F16Op:$rd, GPRV2F16Op:$rs, GPRV2F16Op:$rs), 0>;
}

//===----------------------------------------------------------------------===//
// Code generation patterns
//===----------------------------------------------------------------------===//

// A splat (both lanes the same scalar) folds into the .r form.
def cv_hv_splat
    : PatFrag<(ops node:$s), (v2f16 (build_vector (f16 node:$s), (f16 node:$s)))>;

multiclass CVVFBinPats<SDPatternOperator Op, string Inst> {
  def : Pat<(v2f16 (Op GPRV2F16:$rs1, GPRV2F16:$rs2)),
            (!cast<Instruction>(Inst # "_H") GPRV2F16:$rs1, GPRV2F16:$rs2)>;
  def : Pat<(v2f16 (Op GPRV2F16:$rs1, (cv_hv_splat FPR16INX:$rs2))),
            (!cast<Instruction>(Inst # "_R_H") GPRV2F16:$rs1, FPR16INX:$rs2)>;
}

let Predicates = [HasVendorXCVhv, IsRV32] in {
// The commuted .r forms (splat on the left of a commutative node) are
// generated by TableGen.
defm : CVVFBinPats<fadd, "CV_VFADD">;
defm : CVVFBinPats<fsub, "CV_VFSUB">;
defm : CVVFBinPats<fmul, "CV_VFMUL">;
defm : CVVFBinPats<fminnum, "CV_VFMIN">;
defm : CVVFBinPats<fmaxnum, "CV_VFMAX">;
defm : CVVFBinPats<fminimumnum, "CV_VFMIN">;
defm : CVVFBinPats<fmaximumnum, "CV_VFMAX">;
// fdiv and fsqrt are not selected: they are scalarized onto fdiv.h / fsqrt.h
// (see RISCVISelLowering.cpp).

// Sign operations only move the sign bits.
def : Pat<(v2f16 (fneg GPRV2F16:$rs1)),
          (CV_VFSGNJN_H GPRV2F16:$rs1, GPRV2F16:$rs1)>;
def : Pat<(v2f16 (fabs GPRV2F16:$rs1)),
          (CV_VFSGNJX_H GPRV2F16:$rs1, GPRV2F16:$rs1)>;
def : Pat<(v2f16 (fcopysign GPRV2F16:$rs1, (v2f16 GPRV2F16:$rs2))),
          (CV_VFSGNJ_H GPRV2F16:$rs1, GPRV2F16:$rs2)>;
def : Pat<(v2f16 (fcopysign GPRV2F16:$rs1, (v2f16 (fneg GPRV2F16:$rs2)))),
          (CV_VFSGNJN_H GPRV2F16:$rs1, GPRV2F16:$rs2)>;

// Fused multiply-add. vfmac rd, rs1, rs2 is rd + rs1*rs2 (tied accumulator).
// vfmre rd, rs1, rs2 is rs1*rs2 - rd, so it is selected for a*b - c.
def : Pat<(v2f16 (fma GPRV2F16:$rs1, GPRV2F16:$rs2, GPRV2F16:$rd)),
          (CV_VFMAC_H GPRV2F16:$rd, GPRV2F16:$rs1, GPRV2F16:$rs2)>;
def : Pat<(v2f16 (fma GPRV2F16:$rs1, (cv_hv_splat FPR16INX:$rs2),
                      GPRV2F16:$rd)),
          (CV_VFMAC_R_H GPRV2F16:$rd, GPRV2F16:$rs1, FPR16INX:$rs2)>;
def : Pat<(v2f16 (fma GPRV2F16:$rs1, GPRV2F16:$rs2,
                      (v2f16 (fneg GPRV2F16:$rd)))),
          (CV_VFMRE_H GPRV2F16:$rd, GPRV2F16:$rs1, GPRV2F16:$rs2)>;
def : Pat<(v2f16 (fma GPRV2F16:$rs1, (cv_hv_splat FPR16INX:$rs2),
                      (v2f16 (fneg GPRV2F16:$rd)))),
          (CV_VFMRE_R_H GPRV2F16:$rd, GPRV2F16:$rs1, FPR16INX:$rs2)>;

// Two floats rounded into one register (lane 0 from rs1).
def : Pat<(v2f16 (build_vector (f16 (fpround (f32 FPR32INX:$rs1))),
                               (f16 (fpround (f32 FPR32INX:$rs2))))),
          (CV_VFCPKA_H_S FPR32INX:$rs1, FPR32INX:$rs2)>;

// Lane 0 is the low half of the register and needs no instruction; lane 1 is
// shifted down. A run-time lane number is custom-lowered to a shift.
def : Pat<(f16 (vector_extract (v2f16 GPRV2F16:$rs1), 0)),
          (f16 (EXTRACT_SUBREG GPRV2F16:$rs1, sub_16))>;
def : Pat<(f16 (vector_extract (v2f16 GPRV2F16:$rs1), 1)),
          (f16 (EXTRACT_SUBREG
                   (i32 (SRLI (i32 (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR)), 16)),
                   sub_16))>;

// Bitcasts: v2f16, i32 and (with Zfinx) f32 are the same GPR.
def : Pat<(v2f16 (bitconvert (i32 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2F16)>;
def : Pat<(i32 (bitconvert (v2f16 GPRV2F16:$rs1))),
          (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR)>;
// float in a GPR (Zfinx, implied by Zhinx).
def : Pat<(v2f16 (bitconvert (f32 FPR32INX:$rs1))),
          (COPY_TO_REGCLASS
              (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR32INX:$rs1, sub_32),
              GPRV2F16)>;
def : Pat<(f32 (bitconvert (v2f16 GPRV2F16:$rs1))),
          (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR)), sub_32)>;
} // Predicates = [HasVendorXCVhv, IsRV32]

// With P, the integer packed types live in GPR as well.
let Predicates = [HasVendorXCVhv, HasStdExtP, IsRV32] in {
def : Pat<(v2f16 (bitconvert (v2i16 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2F16)>;
def : Pat<(v2i16 (bitconvert (v2f16 GPRV2F16:$rs1))),
          (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR)>;
def : Pat<(v2f16 (bitconvert (v4i8 GPR:$rs1))),
          (COPY_TO_REGCLASS GPR:$rs1, GPRV2F16)>;
def : Pat<(v4i8 (bitconvert (v2f16 GPRV2F16:$rs1))),
          (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR)>;
}

// Without XCVsimd, a splat that did not fold into a .r form: x | x << 16.
def NoVendorXCVsimd : Predicate<"!Subtarget->hasVendorXCVsimd()">;
let Predicates = [HasVendorXCVhv, NoVendorXCVsimd, IsRV32] in
def : Pat<(v2f16 (build_vector (f16 FPR16INX:$s), (f16 FPR16INX:$s))),
          (COPY_TO_REGCLASS
              (OR (SLLI (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16INX:$s,
                                       sub_16), 16),
                  (SRLI (SLLI (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                             FPR16INX:$s, sub_16), 16), 16)),
              GPRV2F16)>;

// Lane moves with XCVsimd. Without XCVsimd, BUILD_VECTOR is custom-lowered to
// integer shift/or sequences instead (RISCVISelLowering.cpp).
//
// NOTE: the low-halves pack is cv.pack, not cv.pack.h:
//   cv.pack   rd, rs1, rs2 : rd[31:16] = rs1[15:0],  rd[15:0] = rs2[15:0]
//   cv.pack.h rd, rs1, rs2 : rd[31:16] = rs1[31:16], rd[15:0] = rs2[31:16]
let Predicates = [HasVendorXCVhv, HasVendorXCVsimd, IsRV32] in {
// Two scalars into one register: lane 0 is the first operand, so it goes to
// the second (low) source of cv.pack. A splat uses one register twice.
def : Pat<(v2f16 (build_vector (f16 FPR16INX:$lo), (f16 FPR16INX:$hi))),
          (COPY_TO_REGCLASS
              (CV_PACK (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16INX:$hi, sub_16),
                       (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16INX:$lo, sub_16)),
              GPRV2F16)>;

// Shuffles and constant-lane inserts reach instruction selection as
// BUILD_VECTORs of lane extracts. cv.shuffle.sci.h takes the lane 0 selector
// from imm bit 0 and the lane 1 selector from bit 1. cv.insert.h rd, rs1, i
// writes rs1[15:0] into lane i of rd and keeps the other lane.
let AddedComplexity = 10 in {
def : Pat<(v2f16 (build_vector (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 1)),
                               (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 0)))),
          (COPY_TO_REGCLASS
              (CV_SHUFFLE_SCI_H (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR), 1),
              GPRV2F16)>;
def : Pat<(v2f16 (build_vector (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 1)),
                               (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 1)))),
          (COPY_TO_REGCLASS
              (CV_SHUFFLE_SCI_H (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR), 3),
              GPRV2F16)>;
def : Pat<(v2f16 (build_vector (f16 (vector_extract (v2f16 GPRV2F16:$rs1), 0)),
                               (f16 (vector_extract (v2f16 GPRV2F16:$rd), 1)))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2F16:$rd, GPR),
                           (COPY_TO_REGCLASS GPRV2F16:$rs1, GPR), 0),
              GPRV2F16)>;
}
def : Pat<(v2f16 (build_vector (f16 FPR16INX:$rs1),
                               (f16 (vector_extract (v2f16 GPRV2F16:$rd), 1)))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2F16:$rd, GPR),
                           (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16INX:$rs1,
                                          sub_16), 0),
              GPRV2F16)>;
def : Pat<(v2f16 (build_vector (f16 (vector_extract (v2f16 GPRV2F16:$rd), 0)),
                               (f16 FPR16INX:$rs1))),
          (COPY_TO_REGCLASS
              (CV_INSERT_H (COPY_TO_REGCLASS GPRV2F16:$rd, GPR),
                           (INSERT_SUBREG (i32 (IMPLICIT_DEF)), FPR16INX:$rs1,
                                          sub_16), 1),
              GPRV2F16)>;
} // Predicates = [HasVendorXCVhv, HasVendorXCVsimd, IsRV32]
