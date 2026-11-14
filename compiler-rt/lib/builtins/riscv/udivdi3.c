//===-- udivdi3.c - Implement __udivdi3 for RV32 with M -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// With the hardware divider, __udivdi3 goes through __udivmoddi4 (see
// riscv/udivmoddi4.c) instead of the generic bit-serial loop.
//
//===----------------------------------------------------------------------===//

#if __riscv_xlen == 32 && defined(__riscv_div)

#include "../int_lib.h"

COMPILER_RT_ABI du_int __udivdi3(du_int a, du_int b) {
  return __udivmoddi4(a, b, (du_int *)0);
}

#else
#include "../udivdi3.c"
#endif
