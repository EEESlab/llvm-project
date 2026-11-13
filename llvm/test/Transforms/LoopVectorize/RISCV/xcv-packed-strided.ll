; RUN: opt < %s -passes=loop-vectorize -mtriple=riscv32 \
; RUN:   -mattr=+m,+zhinx,+xcvhv,+xcvbf,+xcvbfv,+xcvsimd,+xcvmem,+unaligned-scalar-mem \
; RUN:   -riscv-xcvhv-vectorize -S | FileCheck %s

; Packed vectors in GPRs (XCVhv, XCVbfv) with XCVsimd: filling a packed
; vector from loads with an unknown stride costs one load and one lane insert
; per element, and gains nothing over the scalar loop, whose strided accesses
; become post-increment loads. The column-to-row copy of a matrix transpose
; (src[m * N + col] -> dst[row + m], two elements per iteration, src and dst
; possibly overlapping) must stay scalar. A consecutive loop is still
; vectorized.

; CHECK-LABEL: @transpose_col_half(
; CHECK-NOT:   vector.body
; CHECK:       ret void
define void @transpose_col_half(ptr %src, ptr %dst, i32 %N, i32 %pairs, i32 %col, i32 %row) {
entry:
  br label %loop

loop:
  %m = phi i32 [ 0, %entry ], [ %m.next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %r0 = mul nsw i32 %m, %N
  %i0 = add nsw i32 %r0, %col
  %p0 = getelementptr inbounds half, ptr %src, i32 %i0
  %a = load half, ptr %p0, align 2
  %m1 = or disjoint i32 %m, 1
  %r1 = mul nsw i32 %m1, %N
  %i1 = add nsw i32 %r1, %col
  %p1 = getelementptr inbounds half, ptr %src, i32 %i1
  %b = load half, ptr %p1, align 2
  %j0 = add nsw i32 %m, %row
  %q0 = getelementptr inbounds half, ptr %dst, i32 %j0
  store half %a, ptr %q0, align 2
  %j1 = add nsw i32 %m1, %row
  %q1 = getelementptr inbounds half, ptr %dst, i32 %j1
  store half %b, ptr %q1, align 2
  %m.next = add nuw nsw i32 %m, 2
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, %pairs
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @transpose_col_bf16(
; CHECK-NOT:   vector.body
; CHECK:       ret void
define void @transpose_col_bf16(ptr %src, ptr %dst, i32 %N, i32 %pairs, i32 %col, i32 %row) {
entry:
  br label %loop

loop:
  %m = phi i32 [ 0, %entry ], [ %m.next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %r0 = mul nsw i32 %m, %N
  %i0 = add nsw i32 %r0, %col
  %p0 = getelementptr inbounds bfloat, ptr %src, i32 %i0
  %a = load bfloat, ptr %p0, align 2
  %m1 = or disjoint i32 %m, 1
  %r1 = mul nsw i32 %m1, %N
  %i1 = add nsw i32 %r1, %col
  %p1 = getelementptr inbounds bfloat, ptr %src, i32 %i1
  %b = load bfloat, ptr %p1, align 2
  %j0 = add nsw i32 %m, %row
  %q0 = getelementptr inbounds bfloat, ptr %dst, i32 %j0
  store bfloat %a, ptr %q0, align 2
  %j1 = add nsw i32 %m1, %row
  %q1 = getelementptr inbounds bfloat, ptr %dst, i32 %j1
  store bfloat %b, ptr %q1, align 2
  %m.next = add nuw nsw i32 %m, 2
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, %pairs
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: @scale_half(
; CHECK:       vector.body:
; CHECK:         load <2 x half>
; CHECK:         fmul <2 x half>
; CHECK:         store <2 x half>
define void @scale_half(ptr noalias %src, ptr noalias %dst, half %s, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds half, ptr %src, i32 %i
  %a = load half, ptr %p, align 2
  %r = fmul half %a, %s
  %q = getelementptr inbounds half, ptr %dst, i32 %i
  store half %r, ptr %q, align 2
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
