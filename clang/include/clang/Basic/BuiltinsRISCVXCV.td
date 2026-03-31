//==- BuiltinsRISCVXCV.td - RISC-V CORE-V Builtin database    ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CORE-V-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class RISCVXCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_riscv_cv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XCValu extension.
//===----------------------------------------------------------------------===//
def alu_sle   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_sleu  : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvalu">;
def alu_exths : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_exthz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;
def alu_extbs : RISCVXCVBuiltin<"int(int)", "xcvalu">;
def alu_extbz : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;

def alu_clip   : RISCVXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_clipu  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvalu">;
def alu_addN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_addRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subN   : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subRN  : RISCVXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
} // Attributes = [NoThrow, Const]

//===----------------------------------------------------------------------===//
// XCVelw extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow] in {
def elw_elw : RISCVXCVBuiltin<"int(int *)", "xcvelw">;
} // Attributes = [NoThrow]

//===----------------------------------------------------------------------===//
// XCVmac extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {
def mac_mac      : RISCVXCVBuiltin<"int(int, int, int)", "xcvmac">;
def mac_msu      : RISCVXCVBuiltin<"int(int, int, int)", "xcvmac">;
def mac_muluN    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulsN    : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhsN  : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_muluRN   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulsRN   : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_mulhhsRN : RISCVXCVBuiltin<"int(unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macuN    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhuN  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macsN    : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhsN  : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macuRN   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhuRN : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_macsRN   : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
def mac_machhsRN : RISCVXCVBuiltin<"int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvmac">;
} // Attributes = [NoThrow, Const]

//===----------------------------------------------------------------------===//
// XCVsimd extension.
//===----------------------------------------------------------------------===//
let Attributes = [NoThrow, Const] in {

// ADD/SUB — halfword has div immediate; byte is plain
def simd_add_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_add_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_add_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_add_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sub_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_sub_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sub_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sub_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// AVG (signed)
def simd_avg_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avg_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avg_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avg_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// AVGU (unsigned)
def simd_avgu_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avgu_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avgu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_avgu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// MIN / MINU / MAX / MAXU
def simd_min_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_min_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_min_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_min_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_minu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_max_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_maxu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// SRL / SRA / SLL (shifts — immediate versions take uimm6)
def simd_srl_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_srl_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_srl_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_srl_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sra_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sra_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sra_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sra_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sll_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sll_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sll_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_sll_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// OR / XOR / AND
def simd_or_h    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_or_b    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_or_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_or_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_xor_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_and_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_and_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_and_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_and_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// ABS
def simd_abs_h   : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;
def simd_abs_b   : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;

// DOT PRODUCTS
def simd_dotup_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotup_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotusp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;
def simd_dotsp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int)", "xcvsimd">;

// SDOT (accumulating dot products — rd is both input and output)
def simd_sdotup_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotup_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_sdotusp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotusp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_sc_h : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;
def simd_sdotsp_sc_b : RISCVXCVBuiltin<"int(unsigned int, unsigned int, int)", "xcvsimd">;

// EXTRACT / INSERT
def simd_extract_h  : RISCVXCVBuiltin<"int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extract_b  : RISCVXCVBuiltin<"int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extractu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_extractu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_insert_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_insert_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;

// SHUFFLE / SHUFFLE2
def simd_shuffle_h    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle_b    : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle_sci_h : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_shuffle_sci_b : RISCVXCVBuiltin<"unsigned int(unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_shuffle2_h   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_shuffle2_b   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;

// PACK
def simd_packhi_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_packlo_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_packhi_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;
def simd_packlo_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvsimd">;

// COMPARE (result is mask in packed reg)
def simd_cmpeq_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpeq_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpeq_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpeq_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpne_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpne_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpne_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpne_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgt_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgt_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgt_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgt_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpge_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpge_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpge_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpge_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmplt_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmplt_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmplt_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmplt_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmple_h  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmple_b  : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmple_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmple_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgtu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgtu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgtu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgtu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgeu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgeu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgeu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpgeu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpltu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpltu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpltu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpltu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpleu_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpleu_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpleu_sc_h : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;
def simd_cmpleu_sc_b : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvsimd">;

// COMPLEX NUMBER OPS
def simd_cplxmul_r : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_cplxmul_i : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;
def simd_cplxconj   : RISCVXCVBuiltin<"unsigned int(unsigned int)", "xcvsimd">;
def simd_subrotmj   : RISCVXCVBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)", "xcvsimd">;

} // Attributes = [NoThrow, Const]
