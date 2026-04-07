// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvmac -target-feature +xcvsimd \
// RUN:   -target-feature +xcvbitmanip -fsyntax-only -verify %s

// This file tests that Sema range checks correctly reject out-of-range
// immediate arguments for XCV builtins.

#include <stdint.h>

// ===== XCVmac — shift must be in [0, 31] =====

void test_mac_range(uint32_t a, uint32_t b, uint32_t c) {
  (void)__builtin_riscv_cv_mac_muluN(a, b, 31);  // OK: max value
  (void)__builtin_riscv_cv_mac_muluN(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  (void)__builtin_riscv_cv_mac_mulsN(a, b, 0);   // OK: min value
  (void)__builtin_riscv_cv_mac_macuN(a, b, c, 31);  // OK
  (void)__builtin_riscv_cv_mac_macuN(a, b, c, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}
