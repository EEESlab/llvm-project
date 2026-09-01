# REQUIRES: riscv
## R_RISCV_CVPCREL_UI12 is a CORE-V vendor relocation (224). It must be mapped
## to a RelExpr in scanSectionImpl, not only in getRelExpr; otherwise the
## scanner hits its default label and reports
##   error: unknown relocation (224) against symbol
##
## The relocation patches the 12-bit unsigned field at bits[31:20] of cv.setup
## with (S - P) >> 2. The instruction is written as a raw word so the test does
## not depend on assembler support for the cv.* mnemonics.

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=RELOC
# RUN: ld.lld %t.o --section-start=.text=0x100000 -o %t
# RUN: llvm-objdump -s -j .text %t | FileCheck %s --check-prefix=CONTENT

# RELOC:      .rela.text {
# RELOC-NEXT:   0x0 R_RISCV_CVPCREL_UI12 loop_end 0x0
# RELOC-NEXT: }

## _start is at 0x100000, loop_end at 0x100040: (0x100040 - 0x100000) >> 2 == 0x10,
## so the immediate field becomes 0x010 and the word becomes 0x0100c7ab.
# CONTENT:      Contents of section .text:
# CONTENT-NEXT:  100000 abc70001

  .text
  .globl _start
_start:
  .reloc ., R_RISCV_CVPCREL_UI12, loop_end
  .word 0x0000c7ab              # cv.setup 0x1, ra, <loop_end>
  .space 60
loop_end:
  ret
