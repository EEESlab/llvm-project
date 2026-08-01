; Register set assignment across a loop nest.
;
; CV32E40P has two sets. An innermost loop takes set 0, a loop whose children
; are all innermost takes set 1, and anything higher stays a software loop.
;
; RUN: llc -mtriple=riscv32 -mattr=+m,+c,+xcvhwlp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; A two-level nest. The inner setup is inside the outer body, so it re-arms
; set 0 on every outer iteration.

define void @nest_two(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: nest_two:
; CHECK:         .option push
; CHECK-NEXT:    .option norvc
; CHECK-NEXT:    .option norelax
; CHECK-NEXT:    cv.setup{{i?}} 1, {{.*}}[[OUTEREXIT:\.LBB[0-9_]+]]
; CHECK:         cv.setup{{i?}} 0, {{.*}}[[INNEREXIT:\.LBB[0-9_]+]]
; CHECK:       [[INNEREXIT]]:{{.*}}
; CHECK:       [[OUTEREXIT]]:{{.*}}
; CHECK:         .option pop
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %idx = mul nuw nsw i32 %i, 64
  %off = add nuw nsw i32 %idx, %j
  %ps = getelementptr inbounds i32, ptr %src, i32 %off
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %pd = getelementptr inbounds i32, ptr %dst, i32 %off
  store i32 %m, ptr %pd, align 4
  %j.next = add nuw nsw i32 %j, 1
  %j.done = icmp eq i32 %j.next, 64
  br i1 %j.done, label %outer.latch, label %inner

outer.latch:
  %acc = add nuw nsw i32 %i, 7
  %pa = getelementptr inbounds i32, ptr %dst, i32 %acc
  store i32 %acc, ptr %pa, align 4
  %i.next = add nuw nsw i32 %i, 1
  %i.done = icmp eq i32 %i.next, 32
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}

; Two sibling innermost loops under one parent. They run in sequence, so they
; share set 0 and the parent still gets set 1.

define void @nest_siblings(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: nest_siblings:
; CHECK:         cv.setup{{i?}} 1,
; CHECK:         cv.setup{{i?}} 0,
; CHECK:         cv.setup{{i?}} 0,
; CHECK-NOT:     cv.setup{{i?}} 1,
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %innerA

innerA:
  %j = phi i32 [ 0, %outer ], [ %j.next, %innerA ]
  %pa = getelementptr inbounds i32, ptr %src, i32 %j
  %va = load i32, ptr %pa, align 4
  %ma = mul i32 %va, 3
  %qa = getelementptr inbounds i32, ptr %dst, i32 %j
  store i32 %ma, ptr %qa, align 4
  %j.next = add nuw nsw i32 %j, 1
  %j.done = icmp eq i32 %j.next, 32
  br i1 %j.done, label %between, label %innerA

between:
  %sep = add nuw nsw i32 %i, 1
  %ps = getelementptr inbounds i32, ptr %dst, i32 %sep
  store i32 %sep, ptr %ps, align 4
  br label %innerB

innerB:
  %k = phi i32 [ 0, %between ], [ %k.next, %innerB ]
  %pb = getelementptr inbounds i32, ptr %src, i32 %k
  %vb = load i32, ptr %pb, align 4
  %mb = mul i32 %vb, 5
  %qb = getelementptr inbounds i32, ptr %dst, i32 %k
  store i32 %mb, ptr %qb, align 4
  %k.next = add nuw nsw i32 %k, 1
  %k.done = icmp eq i32 %k.next, 32
  br i1 %k.done, label %outer.latch, label %innerB

outer.latch:
  %t0 = add nuw nsw i32 %i, 11
  %pt = getelementptr inbounds i32, ptr %dst, i32 %t0
  store i32 %t0, ptr %pt, align 4
  %i.next = add nuw nsw i32 %i, 1
  %i.done = icmp eq i32 %i.next, 16
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}

; Three levels. Only the deepest two can be converted, so the outermost loop
; keeps its software backedge.

define void @nest_three(ptr noalias %dst, ptr noalias %src) {
; CHECK-LABEL: nest_three:
; CHECK:         cv.setup{{i?}} 1,
; CHECK:         cv.setup{{i?}} 0,
; CHECK:         .option pop
; CHECK:         bne
entry:
  br label %l1

l1:
  %i = phi i32 [ 0, %entry ], [ %i.next, %l1.latch ]
  br label %l2

l2:
  %j = phi i32 [ 0, %l1 ], [ %j.next, %l2.latch ]
  br label %l3

l3:
  %k = phi i32 [ 0, %l2 ], [ %k.next, %l3 ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %k
  %v = load i32, ptr %ps, align 4
  %m = mul i32 %v, 3
  %pd = getelementptr inbounds i32, ptr %dst, i32 %k
  store i32 %m, ptr %pd, align 4
  %k.next = add nuw nsw i32 %k, 1
  %k.done = icmp eq i32 %k.next, 16
  br i1 %k.done, label %l2.latch, label %l3

l2.latch:
  %t = add nuw nsw i32 %j, 3
  %pt = getelementptr inbounds i32, ptr %dst, i32 %t
  store i32 %t, ptr %pt, align 4
  %j.next = add nuw nsw i32 %j, 1
  %j.done = icmp eq i32 %j.next, 16
  br i1 %j.done, label %l1.latch, label %l2

l1.latch:
  %i.next = add nuw nsw i32 %i, 1
  %i.done = icmp eq i32 %i.next, 16
  br i1 %i.done, label %exit, label %l1

exit:
  ret void
}

