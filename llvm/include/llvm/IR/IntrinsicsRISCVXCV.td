//===- IntrinsicsRISCVXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the CORE-V vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

class ScalarCoreVBitManipGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVBitManipGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVMacGprGprGprIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

class ScalarCoreVMacGprGPRImmIntrinsic
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

class ScalarCoreVMacGprGprGprImmIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<3>>]>;

// Binary: (i32, i32) -> i32, pure
class ScalarCoreVSimdGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

// Unary: (i32) -> i32, pure
class ScalarCoreVSimdGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

// Ternary: (i32, i32, i32) -> i32, pure (for accumulating ops like sdot)
class ScalarCoreVSimdGprGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

// Binary with immediate at arg2: (i32, i32, IMM) -> i32
// Used for: add_h/sub_h (divcode), subrotmj (divcode)
class ScalarCoreVSimdGprGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<2>>]>;

// Unary with immediate at arg1: (i32, IMM) -> i32
// Used for: extract, extractu, shuffle_sci_h, shuffle_sci_b
class ScalarCoreVSimdGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<1>>]>;

// Ternary with immediate at arg2, where arg0 is rd (accumulator):
// (rd_in, rs1, IMM) -> rd_out
// Used for: insert_h, insert_b
class ScalarCoreVSimdGprGprImm2WbIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<2>>]>;

// Quaternary with immediate at arg3: (rs1, rs2, rd_in, IMM) -> rd_out
// Used for: cplxmul_r, cplxmul_i
class ScalarCoreVSimdGprGprGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<3>>]>;

let TargetPrefix = "riscv" in {
  def int_riscv_cv_bitmanip_extract : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_extractu : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_bclr : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_bset : ScalarCoreVBitManipGprGprIntrinsic;

  def int_riscv_cv_bitmanip_insert
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

  def int_riscv_cv_bitmanip_clb : ScalarCoreVBitManipGprIntrinsic;

  def int_riscv_cv_bitmanip_bitrev
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable,
                            ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

  def int_riscv_cv_alu_clip   : ScalarCoreVAluGprGprIntrinsic;
  def int_riscv_cv_alu_clipu  : ScalarCoreVAluGprGprIntrinsic;
  def int_riscv_cv_alu_addN   : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_adduN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_addRN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_adduRN : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subN   : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subuN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subRN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subuRN : ScalarCoreVAluGprGprGprIntrinsic;

  def int_riscv_cv_mac_mac : ScalarCoreVMacGprGprGprIntrinsic;
  def int_riscv_cv_mac_msu : ScalarCoreVMacGprGprGprIntrinsic;

  def int_riscv_cv_mac_muluN    : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhuN  : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulsN    : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhsN  : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_muluRN   : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhuRN : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulsRN   : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhsRN : ScalarCoreVMacGprGPRImmIntrinsic;

  def int_riscv_cv_mac_macuN    : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhuN  : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macsN    : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhsN  : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macuRN   : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhuRN : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macsRN   : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhsRN : ScalarCoreVMacGprGprGprImmIntrinsic;

  def int_riscv_cv_elw_elw
    : Intrinsic<[llvm_i32_ty], [llvm_ptr_ty],
                [IntrReadMem, IntrArgMemOnly, IntrHasSideEffects]>;

  // ==== XCVsimd — ADD/SUB ====
  // add_h takes a divcode immediate (0=none, 1=/2, 2=/4, 3=/8)
  def int_riscv_cv_simd_add_h     : ScalarCoreVSimdGprGprImmIntrinsic;
  def int_riscv_cv_simd_add_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_add_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_add_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sub_h     : ScalarCoreVSimdGprGprImmIntrinsic;
  def int_riscv_cv_simd_sub_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sub_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sub_sc_b  : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — AVG / AVGU ====
  def int_riscv_cv_simd_avg_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avg_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avg_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avg_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avgu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avgu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avgu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_avgu_sc_b : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — MIN / MINU / MAX / MAXU ====
  def int_riscv_cv_simd_min_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_min_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_min_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_min_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_minu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_minu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_minu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_minu_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_max_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_max_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_max_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_max_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_maxu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_maxu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_maxu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_maxu_sc_b : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — SRL / SRA / SLL ====
  def int_riscv_cv_simd_srl_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_srl_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_srl_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_srl_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sra_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sra_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sra_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sra_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sll_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sll_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sll_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_sll_sc_b  : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — OR / XOR / AND ====
  def int_riscv_cv_simd_or_h      : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_or_b      : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_or_sc_h   : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_or_sc_b   : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_xor_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_xor_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_xor_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_xor_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_and_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_and_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_and_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_and_sc_b  : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — ABS ====
  def int_riscv_cv_simd_abs_h     : ScalarCoreVSimdGprIntrinsic;
  def int_riscv_cv_simd_abs_b     : ScalarCoreVSimdGprIntrinsic;

  // ==== XCVsimd — DOT PRODUCTS (non-accumulating) ====
  def int_riscv_cv_simd_dotup_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotup_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotup_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotup_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotusp_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotusp_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotusp_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotusp_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotsp_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotsp_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotsp_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_dotsp_sc_b : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — SDOT (accumulating dot products): (rs1, rs2, acc) -> acc ====
  def int_riscv_cv_simd_sdotup_h    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotup_b    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotup_sc_h : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotup_sc_b : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotusp_h    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotusp_b    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotusp_sc_h : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotusp_sc_b : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotsp_h    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotsp_b    : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotsp_sc_h : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_sdotsp_sc_b : ScalarCoreVSimdGprGprGprIntrinsic;

  // ==== XCVsimd — EXTRACT / EXTRACTU: (rs1, IMM) -> rd ====
  def int_riscv_cv_simd_extract_h  : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_extract_b  : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_extractu_h : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_extractu_b : ScalarCoreVSimdGprImmIntrinsic;

  // ==== XCVsimd — INSERT: (rd_in, rs1, IMM) -> rd_out ====
  def int_riscv_cv_simd_insert_h   : ScalarCoreVSimdGprGprImm2WbIntrinsic;
  def int_riscv_cv_simd_insert_b   : ScalarCoreVSimdGprGprImm2WbIntrinsic;

  // ==== XCVsimd — SHUFFLE ====
  def int_riscv_cv_simd_shuffle_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_shuffle_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_shuffle_sci_h : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_shuffle_sci_b : ScalarCoreVSimdGprImmIntrinsic;

  // ==== XCVsimd — SHUFFLE2: (rs1, rs2, rd_in) -> rd_out ====
  def int_riscv_cv_simd_shuffle2_h : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_shuffle2_b : ScalarCoreVSimdGprGprGprIntrinsic;

  // ==== XCVsimd — PACK ====
  // pack / pack.h: (rs1, rs2) -> rd
  def int_riscv_cv_simd_packlo_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_packhi_h : ScalarCoreVSimdGprGprIntrinsic;
  // packhi.b / packlo.b: (rd_in, rs1, rs2) -> rd_out  (read-modify-write)
  def int_riscv_cv_simd_packhi_b : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_packlo_b : ScalarCoreVSimdGprGprGprIntrinsic;

  // ==== XCVsimd — COMPARE ====
  def int_riscv_cv_simd_cmpeq_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpeq_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpeq_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpeq_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpne_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpne_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpne_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpne_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgt_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgt_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgt_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgt_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpge_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpge_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpge_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpge_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmplt_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmplt_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmplt_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmplt_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmple_h     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmple_b     : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmple_sc_h  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmple_sc_b  : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgtu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgtu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgtu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgtu_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgeu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgeu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgeu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpgeu_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpltu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpltu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpltu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpltu_sc_b : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpleu_h    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpleu_b    : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpleu_sc_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_cmpleu_sc_b : ScalarCoreVSimdGprGprIntrinsic;

  // ==== XCVsimd — COMPLEX ====
  // cplxmul: (rs1, rs2, rd_in, divcode_IMM) -> rd_out
  def int_riscv_cv_simd_cplxmul_r : ScalarCoreVSimdGprGprGprImmIntrinsic;
  def int_riscv_cv_simd_cplxmul_i : ScalarCoreVSimdGprGprGprImmIntrinsic;
  def int_riscv_cv_simd_cplxconj  : ScalarCoreVSimdGprIntrinsic;
  // subrotmj: (rs1, rs2, divcode_IMM) -> rd
  def int_riscv_cv_simd_subrotmj  : ScalarCoreVSimdGprGprImmIntrinsic;
} // TargetPrefix = "riscv"
