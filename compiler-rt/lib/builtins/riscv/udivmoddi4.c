//===-- udivmoddi4.c - Implement __udivmoddi4 for RV32 with M -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __udivmoddi4 for RV32 targets that have the hardware
// divider (the M extension, __riscv_div). The generic implementation only
// uses the divider when the dividend fits in 32 bits and otherwise produces
// the quotient one bit per loop iteration. Here every 64-bit division is
// reduced to at most three 32-bit hardware divisions, each with its
// remainder (schoolbook long division on normalized operands, Knuth's
// Algorithm D with 16-bit digits for the 64-by-32 step).
//
//===----------------------------------------------------------------------===//

#if __riscv_xlen == 32 && defined(__riscv_div)

#include "../int_lib.h"

// Divide the two-word value u1:u0 by v and return the quotient; the remainder
// goes to *r. Requires v to be normalized (top bit set) and u1 < v, so the
// quotient fits in one word. Two 32-by-16 hardware divisions.
static __inline su_int __udiv64by32(su_int u1, su_int u0, su_int v,
                                    su_int *r) {
  const su_int b = 1U << 16;
  su_int vn1 = v >> 16;
  su_int vn0 = v & 0xFFFF;
  su_int un1 = u0 >> 16;
  su_int un0 = u0 & 0xFFFF;

  // First quotient digit, from u1:un1 / v. The estimate from the top digit of
  // v is at most 2 too large.
  su_int q1 = u1 / vn1;
  su_int rhat = u1 % vn1;
  while (q1 >= b || q1 * vn0 > ((rhat << 16) | un1)) {
    --q1;
    rhat += vn1;
    if (rhat >= b)
      break;
  }
  // Wraps modulo 2^32, which is fine: the true value fits in 32 bits.
  su_int un21 = ((u1 << 16) | un1) - q1 * v;

  // Second quotient digit, from un21:un0 / v.
  su_int q0 = un21 / vn1;
  rhat = un21 % vn1;
  while (q0 >= b || q0 * vn0 > ((rhat << 16) | un0)) {
    --q0;
    rhat += vn1;
    if (rhat >= b)
      break;
  }

  *r = ((un21 << 16) | un0) - q0 * v;
  return (q1 << 16) | q0;
}

COMPILER_RT_ABI du_int __udivmoddi4(du_int a, du_int b, du_int *rem) {
  const unsigned n_uword_bits = sizeof(su_int) * CHAR_BIT;
  udwords n, d, q, r;
  n.all = a;
  d.all = b;
  su_int n1 = n.s.high, n0 = n.s.low;
  su_int d1 = d.s.high, d0 = d.s.low;

  if (d1 == 0) {
    if (d0 == 0) {
      // Division by zero is unspecified. Do as the RISC-V divider does: the
      // quotient is all ones and the remainder is the dividend.
      if (rem)
        *rem = a;
      return ~(du_int)0;
    }
    if (n1 == 0) {
      // 32 / 32: one hardware division.
      if (rem)
        *rem = n0 % d0;
      return n0 / d0;
    }
    // 64 / 32. The high quotient word is n1 / d0; its remainder is the high
    // word of the second, two-word-by-one-word, division.
    q.s.high = 0;
    if (n1 >= d0) {
      q.s.high = n1 / d0;
      n1 %= d0;
    }
    unsigned s = clzsi(d0);
    if (s != 0) {
      d0 <<= s;
      n1 = (n1 << s) | (n0 >> (n_uword_bits - s));
      n0 <<= s;
    }
    su_int r0;
    q.s.low = __udiv64by32(n1, n0, d0, &r0);
    if (rem)
      *rem = r0 >> s;
    return q.all;
  }

  // The divisor has two words: the quotient has one.
  if (d1 > n1) {
    if (rem)
      *rem = a;
    return 0;
  }
  unsigned s = clzsi(d1);
  if (s == 0) {
    // d1 has its top bit set and n1 >= d1: the quotient is 0 or 1.
    if (a >= b) {
      if (rem)
        *rem = a - b;
      return 1;
    }
    if (rem)
      *rem = a;
    return 0;
  }
  // Normalize the divisor and divide the top two words of the shifted
  // dividend by its top word. The estimate is at most 1 too large.
  su_int n2 = n1 >> (n_uword_bits - s);
  n1 = (n1 << s) | (n0 >> (n_uword_bits - s));
  n0 <<= s;
  d1 = (d1 << s) | (d0 >> (n_uword_bits - s));
  d0 <<= s;

  su_int r1;
  su_int q0 = __udiv64by32(n2, n1, d1, &r1);
  udwords m;
  m.all = (du_int)q0 * d0;
  r.s.high = r1;
  r.s.low = n0;
  if (m.all > r.all) {
    --q0;
    d.s.high = d1;
    d.s.low = d0;
    m.all -= d.all;
  }
  if (rem)
    *rem = (r.all - m.all) >> s;
  return q0;
}

#else
#include "../udivmoddi4.c"
#endif
