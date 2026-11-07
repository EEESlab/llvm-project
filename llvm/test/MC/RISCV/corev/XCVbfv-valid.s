# RUN: llvm-mc -triple=riscv32 --mattr=+xcvbfv,+xcvhv -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbfv,+xcvhv < %s \
# RUN:     | llvm-objdump --mattr=+xcvbfv,+xcvhv -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 -mattr=+xcvbf,+xcvhv %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.vfadd.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfadd.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfadd.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfadd.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfadd.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfadd.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfadd.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfadd.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsub.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsub.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsub.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsub.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsub.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsub.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsub.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsub.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmul.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmul.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmul.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmul.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmul.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmul.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmul.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmul.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfdiv.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfdiv.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfdiv.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfdiv.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfdiv.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfdiv.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfdiv.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfdiv.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmin.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmin.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmin.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmin.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmin.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmin.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmin.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmin.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmax.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmax.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmax.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmax.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmax.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmax.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmax.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmax.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsqrt.bf16 a0, a1
# CHECK-INSTR: cv.vfsqrt.bf16 a0, a1
# CHECK-ENCODING: [0x33,0x95,0x05,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsqrt.bf16 t6, s0
# CHECK-INSTR: cv.vfsqrt.bf16 t6, s0
# CHECK-ENCODING: [0xb3,0x1f,0x04,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmac.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmac.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmac.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmac.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmac.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmac.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmac.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmac.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmre.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmre.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmre.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmre.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmre.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfmre.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfmre.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfmre.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfclass.bf16 a0, a1
# CHECK-INSTR: cv.vfclass.bf16 a0, a1
# CHECK-ENCODING: [0x33,0x95,0x15,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfclass.bf16 t6, s0
# CHECK-INSTR: cv.vfclass.bf16 t6, s0
# CHECK-ENCODING: [0xb3,0x1f,0x14,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnj.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnj.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnj.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnj.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnj.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnj.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnj.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnj.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjn.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnjn.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjn.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnjn.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjn.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnjn.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjn.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnjn.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjx.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnjx.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjx.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnjx.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjx.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfsgnjx.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfsgnjx.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfsgnjx.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfeq.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfeq.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfeq.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfeq.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfeq.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfeq.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfeq.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfeq.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfne.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfne.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfne.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfne.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfne.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfne.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfne.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfne.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vflt.bf16 a0, a1, a2
# CHECK-INSTR: cv.vflt.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vflt.bf16 t6, s0, t0
# CHECK-INSTR: cv.vflt.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vflt.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vflt.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vflt.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vflt.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfge.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfge.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfge.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfge.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfge.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfge.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfge.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfge.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfle.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfle.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfle.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfle.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfle.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfle.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfle.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfle.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfgt.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfgt.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfgt.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfgt.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfgt.r.bf16 a0, a1, a2
# CHECK-INSTR: cv.vfgt.r.bf16 a0, a1, a2
# CHECK-ENCODING: [0x33,0xd5,0xc5,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfgt.r.bf16 t6, s0, t0
# CHECK-INSTR: cv.vfgt.r.bf16 t6, s0, t0
# CHECK-ENCODING: [0xb3,0x5f,0x54,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcpka.bf16.s a0, a1, a2
# CHECK-INSTR: cv.vfcpka.bf16.s a0, a1, a2
# CHECK-ENCODING: [0x33,0x95,0xc5,0xb0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcpka.bf16.s t6, s0, t0
# CHECK-INSTR: cv.vfcpka.bf16.s t6, s0, t0
# CHECK-ENCODING: [0xb3,0x1f,0x54,0xb0]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.x.bf16 a0, a1
# CHECK-INSTR: cv.vfcvt.x.bf16 a0, a1
# CHECK-ENCODING: [0x33,0x95,0x25,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.x.bf16 t6, s0
# CHECK-INSTR: cv.vfcvt.x.bf16 t6, s0
# CHECK-ENCODING: [0xb3,0x1f,0x24,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.xu.bf16 a0, a1
# CHECK-INSTR: cv.vfcvt.xu.bf16 a0, a1
# CHECK-ENCODING: [0x33,0xd5,0x25,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.xu.bf16 t6, s0
# CHECK-INSTR: cv.vfcvt.xu.bf16 t6, s0
# CHECK-ENCODING: [0xb3,0x5f,0x24,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.x a0, a1
# CHECK-INSTR: cv.vfcvt.bf16.x a0, a1
# CHECK-ENCODING: [0x33,0x95,0x35,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.x t6, s0
# CHECK-INSTR: cv.vfcvt.bf16.x t6, s0
# CHECK-ENCODING: [0xb3,0x1f,0x34,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.xu a0, a1
# CHECK-INSTR: cv.vfcvt.bf16.xu a0, a1
# CHECK-ENCODING: [0x33,0xd5,0x35,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.xu t6, s0
# CHECK-INSTR: cv.vfcvt.bf16.xu t6, s0
# CHECK-ENCODING: [0xb3,0x5f,0x34,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.h.bf16 a0, a1
# CHECK-INSTR: cv.vfcvt.h.bf16 a0, a1
# CHECK-ENCODING: [0x33,0xa5,0x55,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.h.bf16 t6, s0
# CHECK-INSTR: cv.vfcvt.h.bf16 t6, s0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.h a0, a1
# CHECK-INSTR: cv.vfcvt.bf16.h a0, a1
# CHECK-ENCODING: [0x33,0x95,0x65,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfcvt.bf16.h t6, s0
# CHECK-INSTR: cv.vfcvt.bf16.h t6, s0
# CHECK-ENCODING: [0xb3,0x1f,0x64,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfneg.bf16 a0, a1
# CHECK-INSTR: cv.vfsgnjn.bf16 a0, a1, a1
# CHECK-ENCODING: [0x33,0x95,0xb5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}

cv.vfabs.bf16 a0, a1
# CHECK-INSTR: cv.vfsgnjx.bf16 a0, a1, a1
# CHECK-ENCODING: [0x33,0x95,0xb5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVbfv' (CORE-V Packed BFloat16 Floating-Point){{$}}
