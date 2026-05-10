; NOTE: Do not use update_llc_test_checks.py on this file.
; The CHECK-NOT directives are intentional negative assertions.
; RUN: llc -mtriple=riscv32 -mattr=+m,+xcvalu -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

;===----------------------------------------------------------------------===;
; Negative tests: Generic XCValu patterns must NOT fire when the inner
; add/sub has more than one user, otherwise the fused cv.{add,sub}{,u}n{,r}
; would clobber the original value of (a+b) / (a-b) which is still needed
; elsewhere in the DAG.
;===----------------------------------------------------------------------===;

; (a + b) used twice: once shifted (would be cv.addn) and once raw.
; The fused pattern must bail out — emit a plain ADD + SRAI instead.
define i32 @test_addn_no_oneuse(i32 %a, i32 %b, ptr %sink) {
; CHECK-LABEL: test_addn_no_oneuse:
; CHECK:       add     [[T:[as][0-9]+]], a0, a1
; CHECK-NOT:   cv.addn
; CHECK:       srai    {{[as][0-9]+}}, [[T]], 3
; CHECK:       sw      [[T]], 0([[SINK:a[0-9]+]])
  %sum   = add i32 %a, %b
  %shift = ashr i32 %sum, 3
  ; Second use of %sum: stored to memory. Forces hasOneUse() to fail.
  store i32 %sum, ptr %sink, align 4
  ret i32 %shift
}

; Same shape with logical shift -> would have been cv.addun.
define i32 @test_addun_no_oneuse(i32 %a, i32 %b, ptr %sink) {
; CHECK-LABEL: test_addun_no_oneuse:
; CHECK:       add     [[T:[as][0-9]+]], a0, a1
; CHECK-NOT:   cv.addun
; CHECK:       srli    {{[as][0-9]+}}, [[T]], 4
  %sum   = add i32 %a, %b
  %shift = lshr i32 %sum, 4
  store i32 %sum, ptr %sink, align 4
  ret i32 %shift
}

; (a - b) reused -> cv.subn must NOT fire.
define i32 @test_subn_no_oneuse(i32 %a, i32 %b, ptr %sink) {
; CHECK-LABEL: test_subn_no_oneuse:
; CHECK:       sub     [[T:[as][0-9]+]], a0, a1
; CHECK-NOT:   cv.subn
; CHECK:       srai    {{[as][0-9]+}}, [[T]], 2
  %diff  = sub i32 %a, %b
  %shift = ashr i32 %diff, 2
  store i32 %diff, ptr %sink, align 4
  ret i32 %shift
}

; NR-format: (rd + rs1) >> rs2 where (rd + rs1) is reused.
; cv.addnr ties rd in/out, so reusing the sum elsewhere is incompatible.
define i32 @test_addnr_no_oneuse(i32 %rd, i32 %rs1, i32 %rs2, ptr %sink) {
; CHECK-LABEL: test_addnr_no_oneuse:
; CHECK:       add     [[T:[as][0-9]+]], a0, a1
; CHECK-NOT:   cv.addnr
; CHECK:       sra     {{[as][0-9]+}}, [[T]], a2
  %sum   = add i32 %rd, %rs1
  %shift = ashr i32 %sum, %rs2
  store i32 %sum, ptr %sink, align 4
  ret i32 %shift
}

;===----------------------------------------------------------------------===;
; Sanity: when the inner add/sub IS single-use, the fused pattern still
; fires correctly. This guarantees the hasOneUse guard is not over-aggressive.
;===----------------------------------------------------------------------===;

define i32 @test_addn_single_use(i32 %a, i32 %b) {
; CHECK-LABEL: test_addn_single_use:
; CHECK:       cv.addn a0, a0, a1, 3
; CHECK-NEXT:  ret
  %sum   = add i32 %a, %b
  %shift = ashr i32 %sum, 3
  ret i32 %shift
}
