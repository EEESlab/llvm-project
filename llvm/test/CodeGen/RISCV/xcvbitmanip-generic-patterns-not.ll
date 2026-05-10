; NOTE: Do not use update_llc_test_checks.py on this file.
; The CHECK-NOT directives are intentional negative assertions.
; RUN: llc -mtriple=riscv32 -mattr=+m,+xcvbitmanip -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

;===----------------------------------------------------------------------===;
; Negative tests: the C++ matchers tryCVBitManipExtractU / Extract / BClr /
; BSet / Insert must bail out when the inner shift/and/or has more than one
; consumer. Otherwise the fused cv.* instruction would compute the right
; value but the *other* consumer would be left reading a register that has
; already been overwritten (or, worse, a register the allocator never wrote).
;===----------------------------------------------------------------------===;

; (x >> 4) used twice: once anded (would be cv.extractu), once stored raw
; to MMIO. cv.extractu would replace the (and (srl x, 4), 7) node but the
; SRL node would still be needed for the store — must NOT fold.
define i32 @test_extractu_no_oneuse(i32 %x, ptr %mmio) {
; CHECK-LABEL: test_extractu_no_oneuse:
; CHECK:       srli    [[S:[as][0-9]+]], a0, 4
; CHECK-NOT:   cv.extractu
; CHECK:       andi    {{[as][0-9]+}}, [[S]], 7
; CHECK:       sw      [[S]], 0(a1)
  %shr   = lshr i32 %x, 4
  %field = and i32 %shr, 7
  ; Second use of %shr: stored to a memory-mapped peripheral register.
  store volatile i32 %shr, ptr %mmio, align 4
  ret i32 %field
}

; (sra (shl x, 24), 24) used twice -> cv.extract must NOT fire.
define i32 @test_extract_no_oneuse(i32 %x, ptr %sink) {
; CHECK-LABEL: test_extract_no_oneuse:
; CHECK:       slli    [[T:[as][0-9]+]], a0, 24
; CHECK-NOT:   cv.extract
; CHECK:       srai    {{[as][0-9]+}}, [[T]], 24
  %shl  = shl i32 %x, 24
  %sext = ashr i32 %shl, 24
  ; Second use of the SHL node forces the matcher to bail out.
  store i32 %shl, ptr %sink, align 4
  ret i32 %sext
}

; (x & ~mask) used in two different downstream computations -> cv.bclr must NOT fire.
; Clearing bits [4:7] of x while also keeping the unmasked computation alive.
define i32 @test_bclr_no_oneuse(i32 %x, ptr %sink, i32 %extra) {
; CHECK-LABEL: test_bclr_no_oneuse:
; CHECK-NOT:   cv.bclr
  %clear = and i32 %x, -241        ; ~0xF0 = 0xFFFFFF0F
  store i32 %clear, ptr %sink, align 4
  ; Second, distinct use of %clear: forces hasOneUse() == false on the AND.
  %sum = add i32 %clear, %extra
  ret i32 %sum
}

; (or x, mask) used in two different downstream computations -> cv.bset must NOT fire.
define i32 @test_bset_no_oneuse(i32 %x, ptr %sink, i32 %extra) {
; CHECK-LABEL: test_bset_no_oneuse:
; CHECK-NOT:   cv.bset
  %set = or i32 %x, 240            ; bits [4:7] = 1
  store i32 %set, ptr %sink, align 4
  %sum = add i32 %set, %extra
  ret i32 %sum
}

; cv.insert idiom but the cleared value (reg & ~(mask<<offset)) is also
; consumed elsewhere — fold must bail out.
define i32 @test_insert_no_oneuse(i32 %reg, i32 %val, ptr %sink) {
; CHECK-LABEL: test_insert_no_oneuse:
; CHECK-NOT:   cv.insert
  %clear  = and i32 %reg, -3841    ; ~(0xF << 8) = 0xFFFFF0FF
  store i32 %clear, ptr %sink, align 4
  %low    = and i32 %val, 15
  %shf    = shl i32 %low, 8
  %merged = or i32 %clear, %shf
  ret i32 %merged
}

;===----------------------------------------------------------------------===;
; Sanity: single-use cases still match the fused instruction.
;===----------------------------------------------------------------------===;

define i32 @test_extractu_single_use(i32 %x) {
; CHECK-LABEL: test_extractu_single_use:
; CHECK:       cv.extractu a0, a0, 2, 4
; CHECK-NEXT:  ret
  %shr   = lshr i32 %x, 4
  %field = and i32 %shr, 7
  ret i32 %field
}
