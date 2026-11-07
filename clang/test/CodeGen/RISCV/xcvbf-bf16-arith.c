// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +zfinx -target-feature +xcvbf \
// RUN:   -target-feature +xcvbfv -O1 -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,XCVBF
// RUN: %clang_cc1 -triple riscv32 -target-feature +zfinx \
// RUN:   -O1 -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,NOXCVBF
// RUN: %clang_cc1 -triple riscv32 -target-feature +zfinx -target-feature +xcvbf \
// RUN:   -E -dM %s -o - | FileCheck %s --check-prefix=MACROS

// With XCVbf, __bf16 arithmetic is native: one bfloat operation per C
// operator, no excess precision. Without it, clang computes in float.

// MACROS: #define __riscv_xcvbf 1000000{{$}}

typedef __bf16 v2bf __attribute__((vector_size(4)));

// CHECK-LABEL: define dso_local bfloat @add(
// XCVBF: fadd bfloat
// XCVBF-NOT: fpext
// NOXCVBF: fpext bfloat {{.*}} to float
// NOXCVBF: fadd float
// NOXCVBF: fptrunc float {{.*}} to bfloat
__bf16 add(__bf16 a, __bf16 b) { return a + b; }

// Each operator rounds to bfloat16.
// CHECK-LABEL: define dso_local bfloat @mul_add(
// XCVBF: [[M:%.*]] = fmul bfloat
// XCVBF: fadd bfloat [[M]]
// NOXCVBF: fmul float
// NOXCVBF-NOT: fptrunc
// NOXCVBF: fadd float
// NOXCVBF: fptrunc float {{.*}} to bfloat
__bf16 mul_add(__bf16 a, __bf16 b, __bf16 c) { return a * b + c; }

// CHECK-LABEL: define dso_local {{.*}}i1 @less(
// XCVBF: fcmp olt bfloat
_Bool less(__bf16 a, __bf16 b) { return a < b; }

// CHECK-LABEL: define dso_local i32 @vadd(
// XCVBF: fadd <2 x bfloat>
v2bf vadd(v2bf a, v2bf b) { return a + b; }
