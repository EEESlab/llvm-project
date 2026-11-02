; RUN: opt -mtriple=riscv32 -mattr=+xcvhwlp -passes=hardware-loops -S %s | FileCheck %s

; A loop with a call cannot become a CORE-V hardware loop, so it must not get
; the hardware-loop intrinsics.

declare float @fabs(float)
declare float @llvm.fabs.f32(float)

; CHECK-LABEL: @with_call(
; CHECK-NOT:     llvm.start.loop.iterations
; CHECK-NOT:     llvm.loop.decrement
; CHECK:       ret float
define float @with_call(ptr %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds float, ptr %p, i32 %i
  %v = load float, ptr %gep, align 4
  %a = call float @fabs(float %v)
  %add = fadd float %acc, %a
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret float %add
}

; The intrinsic lowers to an instruction and does not block the conversion.
; CHECK-LABEL: @with_intrinsic(
; CHECK:         llvm.start.loop.iterations
; CHECK:         llvm.loop.decrement
define float @with_intrinsic(ptr %p, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds float, ptr %p, i32 %i
  %v = load float, ptr %gep, align 4
  %a = call float @llvm.fabs.f32(float %v)
  %add = fadd float %acc, %a
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret float %add
}

; The call is in the inner loop: the parent loop is rejected as well.
; CHECK-LABEL: @nested_call(
; CHECK-NOT:     llvm.start.loop.iterations
; CHECK-NOT:     llvm.loop.decrement
; CHECK:       ret void
define void @nested_call(ptr %p, i32 %n) {
entry:
  br label %outer

outer:
  %j = phi i32 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner

inner:
  %i = phi i32 [ 0, %outer ], [ %i.next, %inner ]
  %gep = getelementptr inbounds float, ptr %p, i32 %i
  %v = load float, ptr %gep, align 4
  %a = call float @fabs(float %v)
  store float %a, ptr %gep, align 4
  %i.next = add nuw nsw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %inner, label %outer.latch

outer.latch:
  %j.next = add nuw nsw i32 %j, 1
  %cmp.j = icmp ult i32 %j.next, %n
  br i1 %cmp.j, label %outer, label %exit

exit:
  ret void
}
