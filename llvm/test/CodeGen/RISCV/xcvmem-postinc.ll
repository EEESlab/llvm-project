; NOTE: Do not use update_llc_test_checks.py on this file.
; The CHECK-NOT directives are intentional negative assertions.
; RUN: llc -mtriple=riscv32 -mattr=+m,+xcvmem -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK

;===----------------------------------------------------------------------===;
; Misaligned post-increment must NOT be encoded with an odd simm12 immediate.
; The alignment-aware predicate (simm12_postinc_h) rejects odd offsets, so
; the backend either materializes the offset in a register and uses the
; register-register variant (cv.sh rs2, (rs1), rs3) or falls back to a
; plain sh + addi sequence. Both are acceptable; what we explicitly forbid
; is a literal odd immediate in the post-inc encoding.
;===----------------------------------------------------------------------===;
define ptr @test_sh_misaligned_postinc(ptr %p, i16 %v) {
; CHECK-LABEL: test_sh_misaligned_postinc:
; CHECK-NOT:   cv.sh {{[as][0-9]+}}, ({{[as][0-9]+}}), 1
; CHECK-NOT:   cv.sh {{[as][0-9]+}}, ({{[as][0-9]+}}), -1
  store i16 %v, ptr %p, align 2
  %next = getelementptr i8, ptr %p, i32 1
  ret ptr %next
}

;===----------------------------------------------------------------------===;
; Aligned post-increment matches cv.sh with offset 2 as a literal immediate.
;===----------------------------------------------------------------------===;
define ptr @test_sh_aligned_postinc(ptr %p, i16 %v) {
; CHECK-LABEL: test_sh_aligned_postinc:
; CHECK:       cv.sh a1, (a0), 2
  store i16 %v, ptr %p, align 2
  %next = getelementptr i8, ptr %p, i32 2
  ret ptr %next
}
