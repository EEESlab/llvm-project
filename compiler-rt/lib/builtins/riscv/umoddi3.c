//===-- umoddi3.c - Implement __umoddi3 for RV32 with M -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// With the hardware divider, __umoddi3 goes through __udivmoddi4 (see
// riscv/udivmoddi4.c) instead of the generic bit-serial loop.
//
//===----------------------------------------------------------------------===//

#if __riscv_xlen == 32 && defined(__riscv_div)

#include "../int_lib.h"

COMPILER_RT_ABI du_int __umoddi3(du_int a, du_int b) {
  du_int r;
  __udivmoddi4(a, b, &r);
  return r;
}

#else
#include "../umoddi3.c"
#endif
