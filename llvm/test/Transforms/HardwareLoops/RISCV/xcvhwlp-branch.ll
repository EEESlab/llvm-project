; RUN: opt -mtriple=riscv32 -mattr=+xcvhwlp -passes=hardware-loops -S %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOCMOV
; RUN: opt -mtriple=riscv32 -mattr=+xcvhwlp,+zicond -passes=hardware-loops -S %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CMOV

; The body of a CORE-V hardware loop cannot contain a branch. A select becomes
; one when the core has no conditional move, so such a loop must not get the
; hardware-loop intrinsics: the machine pass would reject it anyway, after LSR
; built a countdown for it.

; Index of the last negative element: the select becomes a branch.
; CHECK-LABEL: @select_index(
; NOCMOV-NOT:    llvm.start.loop.iterations
; CMOV:          llvm.start.loop.iterations
; CHECK:         ret i32
define i32 @select_index(ptr %v, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %r = phi i32 [ -1, %entry ], [ %r.next, %loop ]
  %p = getelementptr inbounds i32, ptr %v, i32 %i
  %x = load i32, ptr %p, align 4
  %neg = icmp slt i32 %x, 0
  %r.next = select i1 %neg, i32 %i, i32 %r
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %r.next
}

; A select of zero lowers to logic operations and is accepted.
; CHECK-LABEL: @select_zero(
; CHECK:         llvm.start.loop.iterations
; CHECK:         ret void
define void @select_zero(ptr %v, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i32, ptr %v, i32 %i
  %x = load i32, ptr %p, align 4
  %neg = icmp slt i32 %x, 0
  %c = select i1 %neg, i32 0, i32 %x
  store i32 %c, ptr %p, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A conditional branch inside the body.
; CHECK-LABEL: @branch_in_body(
; CHECK-NOT:     llvm.start.loop.iterations
; CHECK:         ret void
define void @branch_in_body(ptr %v, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %p = getelementptr inbounds i32, ptr %v, i32 %i
  %x = load i32, ptr %p, align 4
  %neg = icmp slt i32 %x, 0
  br i1 %neg, label %then, label %latch

then:
  store i32 0, ptr %p, align 4
  br label %latch

latch:
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
