; RUN: opt -passes=slp-vectorizer -mtriple=riscv32 -mattr=+m,+xcvbfv,+xcvsimd \
; RUN:   -riscv-xcvhv-vectorize -S < %s | FileCheck %s --check-prefix=VEC
; RUN: opt -passes=slp-vectorizer -mtriple=riscv32 -mattr=+m,+xcvbfv,+xcvsimd \
; RUN:   -S < %s | FileCheck %s --check-prefix=NOVEC

; CORE-V XCVbfv: two adjacent bfloat16 FMAs become one <2 x bfloat>
; operation when the vectorizers may form 32-bit GPR vectors.

define void @axpy2(ptr align 4 %y, ptr align 4 %x, bfloat %a) {
; VEC-LABEL: @axpy2(
; VEC: load <2 x bfloat>
; VEC: load <2 x bfloat>
; VEC: call <2 x bfloat> @llvm.fma.v2bf16
; VEC: store <2 x bfloat>
; NOVEC-LABEL: @axpy2(
; NOVEC-NOT: <2 x bfloat>
  %x1p = getelementptr inbounds bfloat, ptr %x, i32 1
  %y1p = getelementptr inbounds bfloat, ptr %y, i32 1
  %x0 = load bfloat, ptr %x, align 4
  %x1 = load bfloat, ptr %x1p, align 2
  %y0 = load bfloat, ptr %y, align 4
  %y1 = load bfloat, ptr %y1p, align 2
  %r0 = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %x0, bfloat %y0)
  %r1 = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %x1, bfloat %y1)
  store bfloat %r0, ptr %y, align 4
  store bfloat %r1, ptr %y1p, align 2
  ret void
}
