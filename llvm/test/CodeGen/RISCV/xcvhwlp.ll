; Positive tests: a converted loop and the encodings it can use.
;
; RUN: llc -mtriple=riscv32 -mattr=+m,+c,+xcvhwlp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+c,+xcvhwlp -verify-machineinstrs \
; RUN:   -riscv-hwloop-force-long-setup < %s | FileCheck %s --check-prefix=LONG
; RUN: llc -mtriple=riscv32 -mattr=+m,+c,+xcvhwlp -verify-machineinstrs \
; RUN:   -riscv-disable-hwloops < %s | FileCheck %s --check-prefix=OFF

; A constant trip count that fits the 12-bit field of cv.setupi.
;
define void @leaf_imm(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: leaf_imm:
; CHECK:         .p2align 2
; CHECK-NEXT:    .option push
; CHECK-NEXT:    .option norvc
; CHECK-NEXT:    .option norelax
; CHECK-NEXT:    cv.setupi 0, 100, [[EXIT:\.LBB[0-9_]+]]
; CHECK:       [[EXIT]]:{{.*}}
; CHECK:         .option pop
;
; LONG-LABEL: leaf_imm:
; LONG:         .p2align 2
; LONG-NEXT:    .option push
; LONG-NEXT:    .option norvc
; LONG-NEXT:    .option norelax
; LONG-NEXT:    cv.starti 0, [[HDR:\.LBB[0-9_]+]]
; LONG-NEXT:    cv.endi 0, [[EXIT:\.LBB[0-9_]+]]
; LONG-NEXT:    cv.counti 0, 100
; LONG:       [[HDR]]:{{.*}}
;
; OFF-LABEL: leaf_imm:
; OFF-NOT:     cv.setup
; OFF-NOT:     cv.starti
; OFF-NOT:     .option norvc
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %i
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %a = add i32 %m, 1
  %pd = getelementptr inbounds i32, ptr %dst, i32 %i
  store i32 %a, ptr %pd, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A trip count only known at run time uses the register form of cv.setup.

define void @leaf_reg(ptr noalias %dst, ptr noalias %src, i32 %n) {
; CHECK-LABEL: leaf_reg:
; CHECK:         .option norvc
; CHECK-NEXT:    .option norelax
; CHECK-NEXT:    cv.setup 0, {{[a-z0-9]+}}, {{\.LBB[0-9_]+}}
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %i
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %a = add i32 %m, 1
  %pd = getelementptr inbounds i32, ptr %dst, i32 %i
  store i32 %a, ptr %pd, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 4095 is the largest count the immediate form can hold.

define void @count_4095(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: count_4095:
; CHECK:         cv.setupi 0, 4095, {{\.LBB[0-9_]+}}
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %i
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %a = add i32 %m, 1
  %pd = getelementptr inbounds i32, ptr %dst, i32 %i
  store i32 %a, ptr %pd, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 4095
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; 4096 does not fit, so ISel must materialize it and use the register form.

define void @count_4096(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: count_4096:
; CHECK:         lui [[C:[a-z0-9]+]], 1
; CHECK:         cv.setup 0, [[C]], {{\.LBB[0-9_]+}}
; CHECK-NOT:     cv.setupi
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %i
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %a = add i32 %m, 1
  %pd = getelementptr inbounds i32, ptr %dst, i32 %i
  store i32 %a, ptr %pd, align 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, 4096
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

