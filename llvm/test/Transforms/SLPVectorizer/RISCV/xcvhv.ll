; RUN: opt -passes=slp-vectorizer -mtriple=riscv32 -mattr=+m,+xcvhv,+xcvsimd \
; RUN:   -riscv-xcvhv-vectorize -S < %s | FileCheck %s --check-prefix=VEC
; RUN: opt -passes=slp-vectorizer -mtriple=riscv32 -mattr=+m,+xcvhv,+xcvsimd \
; RUN:   -S < %s | FileCheck %s --check-prefix=NOVEC

; CORE-V Xcvhv: two adjacent half FMAs become one <2 x half> operation when
; the vectorizers are allowed to form 32-bit GPR vectors.

define void @axpy2(ptr align 4 %y, ptr align 4 %x, half %a) {
; VEC-LABEL: @axpy2(
; VEC: load <2 x half>
; VEC: load <2 x half>
; VEC: call <2 x half> @llvm.fma.v2f16
; VEC: store <2 x half>
; NOVEC-LABEL: @axpy2(
; NOVEC-NOT: <2 x half>
  %x1p = getelementptr inbounds half, ptr %x, i32 1
  %y1p = getelementptr inbounds half, ptr %y, i32 1
  %x0 = load half, ptr %x, align 4
  %x1 = load half, ptr %x1p, align 2
  %y0 = load half, ptr %y, align 4
  %y1 = load half, ptr %y1p, align 2
  %r0 = call half @llvm.fma.f16(half %a, half %x0, half %y0)
  %r1 = call half @llvm.fma.f16(half %a, half %x1, half %y1)
  store half %r0, ptr %y, align 4
  store half %r1, ptr %y1p, align 2
  ret void
}
