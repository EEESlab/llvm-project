; NOTE: Do not use update_llc_test_checks.py on this file.
; The CHECK-NOT directives are intentional negative assertions.
; RUN: llc -mtriple=riscv32 -mattr=+m,+xcvmac -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; mul result has TWO uses: one in the add, one in a separate compare.
; cv.mac would clobber the multiplication result -> we MUST emit mul + add.
define i32 @test_mac_no_oneuse(i32 %a, i32 %b, i32 %acc) {
; CHECK-LABEL: test_mac_no_oneuse:
; CHECK:       mul     {{a[0-9]}}, a0, a1
; CHECK-NOT:   cv.mac
  %mul = mul i32 %a, %b
  %sum = add i32 %acc, %mul
  %ok  = icmp sgt i32 %mul, 0
  %sel = select i1 %ok, i32 %sum, i32 %mul
  ret i32 %sel
}
