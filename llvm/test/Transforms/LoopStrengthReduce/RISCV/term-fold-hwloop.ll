; RUN: opt -mtriple=riscv32 -mattr=+xcvhwlp -passes=loop-term-fold -S %s \
; RUN:   | FileCheck %s --check-prefix=HWLP
; RUN: opt -mtriple=riscv32 -passes=loop-term-fold -S %s \
; RUN:   | FileCheck %s --check-prefix=NOHWLP

; A loop that will become a CORE-V hardware loop keeps its counter: the
; hardware loop replaces the exit test and takes its count from the original
; trip count. Folding the exit test into a pointer compare would make the
; count a pointer distance, computed as ((end - start) >> 2) + 1.

; HWLP-LABEL: @copy(
; HWLP:         icmp eq i32 %i.next, %n
; HWLP-NOT:     lsr_fold_term_cond
; NOHWLP-LABEL: @copy(
; NOHWLP:       lsr_fold_term_cond.replaced_term_cond
define void @copy(ptr %dst, ptr %src, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = phi ptr [ %src, %entry ], [ %ps.next, %loop ]
  %pd = phi ptr [ %dst, %entry ], [ %pd.next, %loop ]
  %v = load i32, ptr %ps, align 4
  store i32 %v, ptr %pd, align 4
  %ps.next = getelementptr inbounds i8, ptr %ps, i32 4
  %pd.next = getelementptr inbounds i8, ptr %pd, i32 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A loop with a call cannot become a hardware loop: it is folded as usual.
; HWLP-LABEL: @with_call(
; HWLP:         lsr_fold_term_cond.replaced_term_cond
declare void @f(i32)
define void @with_call(ptr %src, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = phi ptr [ %src, %entry ], [ %ps.next, %loop ]
  %v = load i32, ptr %ps, align 4
  call void @f(i32 %v)
  %ps.next = getelementptr inbounds i8, ptr %ps, i32 4
  %i.next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
