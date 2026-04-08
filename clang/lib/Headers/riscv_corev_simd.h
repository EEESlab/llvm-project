/*===---- riscv_corev_simd.h - CORE-V SIMD intrinsics ----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides C intrinsics for the CORE-V XCVsimd ISA extension.
 * Include this header when compiling with -march=..._xcvsimd.
 *
 * All operations work on packed sub-word elements within a 32-bit GPR:
 *   - .h variants operate on two packed int16 / uint16 (halfwords)
 *   - .b variants operate on four packed int8  / uint8  (bytes)
 *   - .sc variants use the lower halfword/byte of a GPR as a scalar operand
 *   - .sci variants use a sign-extended 6-bit immediate as a scalar operand
 *
 * Spec: https://docs.openhwgroup.org/projects/cv32e40p-user-manual/en/latest/
 *       instruction_set_extensions.html#simd
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_COREV_SIMD_H
#define __RISCV_COREV_SIMD_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xcvsimd)

#define __DEFAULT_FN_ATTRS                                                      \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

/* ===========================================================================
 * ADD / SUB
 *
 * .h: cv.add.h  rd, rs1, rs2 (or .div2/.div4/.div8)
 * .b: cv.add.b  rd, rs1, rs2
 *
 * The _h variant takes a div-shift code as third argument (0=no shift,
 * 1=/2, 2=/4, 3=/8). This is an immediate, so it must be a compile-time
 * constant — hence the macro form.
 * =========================================================================== */

/* add.h with div-by-N right shift (DIVCODE in [0,3]) */
#define __riscv_cv_simd_add_h(__rs1, __rs2, __DIVCODE)                         \
  ((uint32_t)__builtin_riscv_cv_simd_add_h((uint32_t)(__rs1),                  \
                                            (uint32_t)(__rs2),                  \
                                            (uint32_t)(__DIVCODE)))

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_add_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_add_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_add_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_add_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_add_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_add_sc_b(a, b);
}

/* sub.h with div-by-N right shift (DIVCODE in [0,3]) */
#define __riscv_cv_simd_sub_h(__rs1, __rs2, __DIVCODE)                         \
  ((uint32_t)__builtin_riscv_cv_simd_sub_h((uint32_t)(__rs1),                  \
                                            (uint32_t)(__rs2),                  \
                                            (uint32_t)(__DIVCODE)))

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sub_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sub_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sub_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sub_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sub_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sub_sc_b(a, b);
}

/* ===========================================================================
 * AVG / AVGU  (signed and unsigned average)
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avg_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avg_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avg_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avg_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avg_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avg_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avg_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avg_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avgu_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avgu_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avgu_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avgu_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avgu_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avgu_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_avgu_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_avgu_sc_b(a, b);
}

/* ===========================================================================
 * MIN / MINU  (signed and unsigned per-element minimum)
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_min_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_min_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_min_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_min_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_minu_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_minu_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_minu_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_minu_sc_b(a, b);
}

/* ===========================================================================
 * MAX / MAXU  (signed and unsigned per-element maximum)
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_max_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_max_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_max_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_max_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_maxu_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_maxu_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_maxu_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_maxu_sc_b(a, b);
}

/* ===========================================================================
 * SHIFT: SRL / SRA / SLL
 * .sc  variants take a GPR shift amount (lower bits used)
 * No .sci variant for shifts (the instruction encodes a small immediate
 * directly in the funct3 space; use the .sc_h/.sc_b with a constant GPR).
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_srl_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_srl_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_srl_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_srl_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_srl_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_srl_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_srl_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_srl_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sra_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sra_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sra_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sra_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sra_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sra_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sra_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sra_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sll_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sll_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sll_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sll_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sll_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sll_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sll_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_sll_sc_b(a, b);
}

/* ===========================================================================
 * BITWISE: OR / XOR / AND
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_or_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_or_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_or_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_or_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_xor_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_xor_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_xor_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_xor_sc_b(a, b);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_and_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_and_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_and_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_and_sc_b(a, b);
}

/* ===========================================================================
 * ABS  (per-element absolute value)
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_abs_h(uint32_t a) {
  return __builtin_riscv_cv_simd_abs_h(a);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_abs_b(uint32_t a) {
  return __builtin_riscv_cv_simd_abs_b(a);
}

/* ===========================================================================
 * DOT PRODUCTS
 *
 * dotup  — unsigned × unsigned dot product  (result is uint32)
 * dotusp — unsigned × signed  dot product  (result is int32)
 * dotsp  — signed   × signed  dot product  (result is int32)
 *
 * sdot* — same but adds result to accumulator register rd (read-modify-write)
 * =========================================================================== */

/* dotup */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotup_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotup_b(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotup_sc_h(a, b);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotup_sc_b(a, b);
}

/* dotusp */
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotusp_h(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotusp_b(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotusp_sc_h(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotusp_sc_b(a, b);
}

/* dotsp */
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotsp_h(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotsp_b(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_sc_h(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotsp_sc_h(a, b);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_sc_b(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_simd_dotsp_sc_b(a, b);
}

/* sdotup (unsigned × unsigned, accumulating) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_h(uint32_t a, uint32_t b, uint32_t acc) {
  return __builtin_riscv_cv_simd_sdotup_h(a, b, acc);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_b(uint32_t a, uint32_t b, uint32_t acc) {
  return __builtin_riscv_cv_simd_sdotup_b(a, b, acc);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_sc_h(uint32_t a, uint32_t b, uint32_t acc) {
  return __builtin_riscv_cv_simd_sdotup_sc_h(a, b, acc);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_sc_b(uint32_t a, uint32_t b, uint32_t acc) {
  return __builtin_riscv_cv_simd_sdotup_sc_b(a, b, acc);
}

/* sdotusp (unsigned × signed, accumulating) */
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_h(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotusp_h(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_b(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotusp_b(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_sc_h(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotusp_sc_h(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_sc_b(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotusp_sc_b(a, b, acc);
}

/* sdotsp (signed × signed, accumulating) */
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_h(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotsp_h(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_b(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotsp_b(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_sc_h(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotsp_sc_h(a, b, acc);
}
static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_sc_b(uint32_t a, uint32_t b, int32_t acc) {
  return __builtin_riscv_cv_simd_sdotsp_sc_b(a, b, acc);
}

/* ===========================================================================
 * EXTRACT / EXTRACTU / INSERT
 *
 * cv.extract.h  rd, rs1, IMM  — extract halfword IMM (0 or 1), sign-extended
 * cv.extract.b  rd, rs1, IMM  — extract byte IMM (0..3), sign-extended
 * cv.extractu.h rd, rs1, IMM  — extract halfword IMM, zero-extended
 * cv.extractu.b rd, rs1, IMM  — extract byte IMM, zero-extended
 * cv.insert.h   rd, rs1, IMM  — insert lower halfword of rs1 into rd at IMM
 * cv.insert.b   rd, rs1, IMM  — insert lower byte of rs1 into rd at IMM
 *
 * IMM is an immediate (compile-time constant), so macros are used.
 * =========================================================================== */

#define __riscv_cv_simd_extract_h(__rs1, __IMM)                                \
  ((int32_t)__builtin_riscv_cv_simd_extract_h((uint32_t)(__rs1),               \
                                               (uint32_t)(__IMM)))

#define __riscv_cv_simd_extract_b(__rs1, __IMM)                                \
  ((int32_t)__builtin_riscv_cv_simd_extract_b((uint32_t)(__rs1),               \
                                               (uint32_t)(__IMM)))

#define __riscv_cv_simd_extractu_h(__rs1, __IMM)                               \
  ((uint32_t)__builtin_riscv_cv_simd_extractu_h((uint32_t)(__rs1),             \
                                                 (uint32_t)(__IMM)))

#define __riscv_cv_simd_extractu_b(__rs1, __IMM)                               \
  ((uint32_t)__builtin_riscv_cv_simd_extractu_b((uint32_t)(__rs1),             \
                                                 (uint32_t)(__IMM)))

/* insert: rd is both input and output (read-modify-write accumulator).
   The intrinsic prototype is (rs1, rd_in, IMM) -> rd_out.               */
#define __riscv_cv_simd_insert_h(__rD, __rs1, __IMM)                           \
  ((uint32_t)__builtin_riscv_cv_simd_insert_h((uint32_t)(__rD),                \
                                               (uint32_t)(__rs1),               \
                                               (uint32_t)(__IMM)))

#define __riscv_cv_simd_insert_b(__rD, __rs1, __IMM)                           \
  ((uint32_t)__builtin_riscv_cv_simd_insert_b((uint32_t)(__rD),                \
                                               (uint32_t)(__rs1),               \
                                               (uint32_t)(__IMM)))

/* ===========================================================================
 * SHUFFLE / SHUFFLE2
 *
 * cv.shuffle.h     rd, rs1, rs2  — shuffle halfwords of rs1 using rs2 mask
 * cv.shuffle.b     rd, rs1, rs2  — shuffle bytes of rs1 using rs2 mask
 * cv.shuffle.sci.h rd, rs1, IMM  — shuffle halfwords using 2-bit imm
 * cv.shuffle.sci.b rd, rs1, IMM  — shuffle bytes using 8-bit imm
 *                                  (split internally into SHUFFLEIx.sci.b)
 * cv.shuffle2.h    rd, rs1, rs2  — interleave shuffle using rs2 mask + rd input
 * cv.shuffle2.b    rd, rs1, rs2  — interleave shuffle using rs2 mask + rd input
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle_h(uint32_t a, uint32_t mask) {
  return __builtin_riscv_cv_simd_shuffle_h(a, mask);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle_b(uint32_t a, uint32_t mask) {
  return __builtin_riscv_cv_simd_shuffle_b(a, mask);
}

/* shuffle.sci.h — IMM is a 2-bit immediate [0,3] */
#define __riscv_cv_simd_shuffle_sci_h(__rs1, __IMM)                            \
  ((uint32_t)__builtin_riscv_cv_simd_shuffle_sci_h((uint32_t)(__rs1),          \
                                                    (uint32_t)(__IMM)))

/* shuffle.sci.b — IMM is an 8-bit immediate [0,255].
   The backend splits this into two bits (top 2 → SHUFFLEIx selector) + 6-bit
   payload, emitting the correct CV_SHUFFLEIx_SCI_B instruction.           */
#define __riscv_cv_simd_shuffle_sci_b(__rs1, __IMM)                            \
  ((uint32_t)__builtin_riscv_cv_simd_shuffle_sci_b((uint32_t)(__rs1),          \
                                                    (uint32_t)(__IMM)))

/* shuffle2: rd is read-modify-write (used as additional input lane) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle2_h(uint32_t rD, uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_shuffle2_h(rs1, rs2, rD);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle2_b(uint32_t rD, uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_shuffle2_b(rs1, rs2, rD);
}

/* ===========================================================================
 * PACK
 *
 * cv.pack    rd, rs1, rs2  — pack lower halfwords: rd = {rs1[15:0], rs2[15:0]}
 * cv.pack.h  rd, rs1, rs2  — pack upper halfwords: rd = {rs1[31:16], rs2[31:16]}
 * cv.packhi.b rd, rs1, rs2 — pack odd bytes:  rd = {rs1[31:24],rs1[15:8],
 *                                                    rs2[31:24],rs2[15:8]}
 * cv.packlo.b rd, rs1, rs2 — pack even bytes: rd = {rs1[23:16],rs1[7:0],
 *                                                    rs2[23:16],rs2[7:0]}
 *
 * packhi/packlo: rd is read-modify-write.
 * =========================================================================== */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packlo_h(uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_packlo_h(rs1, rs2);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packhi_h(uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_packhi_h(rs1, rs2);
}

/* packhi/packlo.b: rD is the accumulator half that keeps unmodified bytes */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packhi_b(uint32_t rD, uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_packhi_b(rD, rs1, rs2);
}
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packlo_b(uint32_t rD, uint32_t rs1, uint32_t rs2) {
  return __builtin_riscv_cv_simd_packlo_b(rD, rs1, rs2);
}

/* ===========================================================================
 * COMPARE  (per-element, result is all-1s or all-0s per lane)
 * =========================================================================== */

/* cmpeq */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpeq_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpeq_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpeq_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpeq_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpeq_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpeq_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpeq_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpeq_sc_b(a, b); }

/* cmpne */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpne_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpne_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpne_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpne_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpne_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpne_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpne_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpne_sc_b(a, b); }

/* cmpgt (signed greater-than) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgt_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgt_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgt_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgt_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgt_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgt_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgt_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgt_sc_b(a, b); }

/* cmpge (signed greater-or-equal) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpge_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpge_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpge_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpge_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpge_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpge_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpge_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpge_sc_b(a, b); }

/* cmplt (signed less-than) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmplt_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmplt_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmplt_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmplt_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmplt_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmplt_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmplt_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmplt_sc_b(a, b); }

/* cmple (signed less-or-equal) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmple_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmple_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmple_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmple_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmple_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmple_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmple_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmple_sc_b(a, b); }

/* cmpgtu (unsigned greater-than) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgtu_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgtu_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgtu_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgtu_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgtu_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgtu_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgtu_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgtu_sc_b(a, b); }

/* cmpgeu (unsigned greater-or-equal) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgeu_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgeu_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgeu_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgeu_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgeu_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgeu_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpgeu_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpgeu_sc_b(a, b); }

/* cmpltu (unsigned less-than) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpltu_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpltu_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpltu_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpltu_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpltu_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpltu_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpltu_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpltu_sc_b(a, b); }

/* cmpleu (unsigned less-or-equal) */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpleu_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpleu_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpleu_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpleu_b(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpleu_sc_h(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpleu_sc_h(a, b); }
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cmpleu_sc_b(uint32_t a, uint32_t b) { return __builtin_riscv_cv_simd_cmpleu_sc_b(a, b); }

/* ===========================================================================
 * COMPLEX NUMBER OPERATIONS
 *
 * cv.cplxmul.r  rd, rs1, rs2 [, divcode]  — complex multiply, real part
 * cv.cplxmul.i  rd, rs1, rs2 [, divcode]  — complex multiply, imaginary part
 *   divcode: 0=no shift, 1=/2, 2=/4, 3=/8  (compile-time constant)
 *   rd is read-modify-write (accumulator for the real or imag lane)
 *
 * cv.cplxconj   rd, rs1       — complex conjugate (negate imaginary part)
 *
 * cv.subrotmj   rd, rs1, rs2 [, divcode]  — subtract + rotate by -j
 *   result = ((rs1 - rs2) >> div) rotated by -j  (swap re/im, negate new im)
 * =========================================================================== */

#define __riscv_cv_simd_cplxmul_r(__rD, __rs1, __rs2, __DIVCODE)               \
  ((uint32_t)__builtin_riscv_cv_simd_cplxmul_r((uint32_t)(__rs1),              \
                                                (uint32_t)(__rs2),              \
                                                (uint32_t)(__rD),               \
                                                (uint32_t)(__DIVCODE)))

#define __riscv_cv_simd_cplxmul_i(__rD, __rs1, __rs2, __DIVCODE)               \
  ((uint32_t)__builtin_riscv_cv_simd_cplxmul_i((uint32_t)(__rs1),              \
                                                (uint32_t)(__rs2),              \
                                                (uint32_t)(__rD),               \
                                                (uint32_t)(__DIVCODE)))

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_cplxconj(uint32_t a) {
  return __builtin_riscv_cv_simd_cplxconj(a);
}

#define __riscv_cv_simd_subrotmj(__rs1, __rs2, __DIVCODE)                      \
  ((uint32_t)__builtin_riscv_cv_simd_subrotmj((uint32_t)(__rs1),               \
                                               (uint32_t)(__rs2),               \
                                               (uint32_t)(__DIVCODE)))

#endif /* __riscv_xcvsimd */

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_COREV_SIMD_H */
