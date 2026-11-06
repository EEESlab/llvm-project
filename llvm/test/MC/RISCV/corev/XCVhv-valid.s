# RUN: llvm-mc -triple=riscv32 --mattr=+xcvhv -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvhv < %s \
# RUN:     | llvm-objdump --mattr=+xcvhv -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 -mattr=+zhinx %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.vfadd.h a0, a1, a2
# CHECK-INSTR: cv.vfadd.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfadd.h t6, s0, t0
# CHECK-INSTR: cv.vfadd.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfadd.r.h a0, a1, a2
# CHECK-INSTR: cv.vfadd.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfadd.r.h t6, s0, t0
# CHECK-INSTR: cv.vfadd.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsub.h a0, a1, a2
# CHECK-INSTR: cv.vfsub.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsub.h t6, s0, t0
# CHECK-INSTR: cv.vfsub.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsub.r.h a0, a1, a2
# CHECK-INSTR: cv.vfsub.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsub.r.h t6, s0, t0
# CHECK-INSTR: cv.vfsub.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x84]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmul.h a0, a1, a2
# CHECK-INSTR: cv.vfmul.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmul.h t6, s0, t0
# CHECK-INSTR: cv.vfmul.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmul.r.h a0, a1, a2
# CHECK-INSTR: cv.vfmul.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmul.r.h t6, s0, t0
# CHECK-INSTR: cv.vfmul.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x86]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfdiv.h a0, a1, a2
# CHECK-INSTR: cv.vfdiv.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfdiv.h t6, s0, t0
# CHECK-INSTR: cv.vfdiv.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfdiv.r.h a0, a1, a2
# CHECK-INSTR: cv.vfdiv.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfdiv.r.h t6, s0, t0
# CHECK-INSTR: cv.vfdiv.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmin.h a0, a1, a2
# CHECK-INSTR: cv.vfmin.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmin.h t6, s0, t0
# CHECK-INSTR: cv.vfmin.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmin.r.h a0, a1, a2
# CHECK-INSTR: cv.vfmin.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmin.r.h t6, s0, t0
# CHECK-INSTR: cv.vfmin.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmax.h a0, a1, a2
# CHECK-INSTR: cv.vfmax.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmax.h t6, s0, t0
# CHECK-INSTR: cv.vfmax.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmax.r.h a0, a1, a2
# CHECK-INSTR: cv.vfmax.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmax.r.h t6, s0, t0
# CHECK-INSTR: cv.vfmax.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x8c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsqrt.h a0, a1
# CHECK-INSTR: cv.vfsqrt.h a0, a1
# CHECK-ENCODING: [0x33,0xa5,0x05,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsqrt.h t6, s0
# CHECK-INSTR: cv.vfsqrt.h t6, s0
# CHECK-ENCODING: [0xb3,0x2f,0x04,0x8e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmac.h a0, a1, a2
# CHECK-INSTR: cv.vfmac.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmac.h t6, s0, t0
# CHECK-INSTR: cv.vfmac.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmac.r.h a0, a1, a2
# CHECK-INSTR: cv.vfmac.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmac.r.h t6, s0, t0
# CHECK-INSTR: cv.vfmac.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmre.h a0, a1, a2
# CHECK-INSTR: cv.vfmre.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmre.h t6, s0, t0
# CHECK-INSTR: cv.vfmre.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmre.r.h a0, a1, a2
# CHECK-INSTR: cv.vfmre.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfmre.r.h t6, s0, t0
# CHECK-INSTR: cv.vfmre.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfclass.h a0, a1
# CHECK-INSTR: cv.vfclass.h a0, a1
# CHECK-ENCODING: [0x33,0xa5,0x15,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfclass.h t6, s0
# CHECK-INSTR: cv.vfclass.h t6, s0
# CHECK-ENCODING: [0xb3,0x2f,0x14,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnj.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnj.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnj.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnj.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnj.r.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnj.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnj.r.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnj.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjn.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnjn.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjn.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnjn.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjn.r.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnjn.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjn.r.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnjn.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjx.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnjx.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjx.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnjx.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjx.r.h a0, a1, a2
# CHECK-INSTR: cv.vfsgnjx.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfsgnjx.r.h t6, s0, t0
# CHECK-INSTR: cv.vfsgnjx.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfeq.h a0, a1, a2
# CHECK-INSTR: cv.vfeq.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfeq.h t6, s0, t0
# CHECK-INSTR: cv.vfeq.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfeq.r.h a0, a1, a2
# CHECK-INSTR: cv.vfeq.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfeq.r.h t6, s0, t0
# CHECK-INSTR: cv.vfeq.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfne.h a0, a1, a2
# CHECK-INSTR: cv.vfne.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfne.h t6, s0, t0
# CHECK-INSTR: cv.vfne.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfne.r.h a0, a1, a2
# CHECK-INSTR: cv.vfne.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfne.r.h t6, s0, t0
# CHECK-INSTR: cv.vfne.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vflt.h a0, a1, a2
# CHECK-INSTR: cv.vflt.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vflt.h t6, s0, t0
# CHECK-INSTR: cv.vflt.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vflt.r.h a0, a1, a2
# CHECK-INSTR: cv.vflt.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vflt.r.h t6, s0, t0
# CHECK-INSTR: cv.vflt.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfge.h a0, a1, a2
# CHECK-INSTR: cv.vfge.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfge.h t6, s0, t0
# CHECK-INSTR: cv.vfge.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfge.r.h a0, a1, a2
# CHECK-INSTR: cv.vfge.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfge.r.h t6, s0, t0
# CHECK-INSTR: cv.vfge.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xa6]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfle.h a0, a1, a2
# CHECK-INSTR: cv.vfle.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfle.h t6, s0, t0
# CHECK-INSTR: cv.vfle.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfle.r.h a0, a1, a2
# CHECK-INSTR: cv.vfle.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfle.r.h t6, s0, t0
# CHECK-INSTR: cv.vfle.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfgt.h a0, a1, a2
# CHECK-INSTR: cv.vfgt.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfgt.h t6, s0, t0
# CHECK-INSTR: cv.vfgt.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfgt.r.h a0, a1, a2
# CHECK-INSTR: cv.vfgt.r.h a0, a1, a2
# CHECK-ENCODING: [0x33,0xe5,0xc5,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfgt.r.h t6, s0, t0
# CHECK-INSTR: cv.vfgt.r.h t6, s0, t0
# CHECK-ENCODING: [0xb3,0x6f,0x54,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcpka.h.s a0, a1, a2
# CHECK-INSTR: cv.vfcpka.h.s a0, a1, a2
# CHECK-ENCODING: [0x33,0xa5,0xc5,0xb0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcpka.h.s t6, s0, t0
# CHECK-INSTR: cv.vfcpka.h.s t6, s0, t0
# CHECK-ENCODING: [0xb3,0x2f,0x54,0xb0]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.x.h a0, a1
# CHECK-INSTR: cv.vfcvt.x.h a0, a1
# CHECK-ENCODING: [0x33,0xa5,0x25,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.x.h t6, s0
# CHECK-INSTR: cv.vfcvt.x.h t6, s0
# CHECK-ENCODING: [0xb3,0x2f,0x24,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.xu.h a0, a1
# CHECK-INSTR: cv.vfcvt.xu.h a0, a1
# CHECK-ENCODING: [0x33,0xe5,0x25,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.xu.h t6, s0
# CHECK-INSTR: cv.vfcvt.xu.h t6, s0
# CHECK-ENCODING: [0xb3,0x6f,0x24,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.h.x a0, a1
# CHECK-INSTR: cv.vfcvt.h.x a0, a1
# CHECK-ENCODING: [0x33,0xa5,0x35,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.h.x t6, s0
# CHECK-INSTR: cv.vfcvt.h.x t6, s0
# CHECK-ENCODING: [0xb3,0x2f,0x34,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.h.xu a0, a1
# CHECK-INSTR: cv.vfcvt.h.xu a0, a1
# CHECK-ENCODING: [0x33,0xe5,0x35,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfcvt.h.xu t6, s0
# CHECK-INSTR: cv.vfcvt.h.xu t6, s0
# CHECK-ENCODING: [0xb3,0x6f,0x34,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfneg.h a0, a1
# CHECK-INSTR: cv.vfsgnjn.h a0, a1, a1
# CHECK-ENCODING: [0x33,0xa5,0xb5,0x9c]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

cv.vfabs.h a0, a1
# CHECK-INSTR: cv.vfsgnjx.h a0, a1, a1
# CHECK-ENCODING: [0x33,0xa5,0xb5,0x9e]
# CHECK-NO-EXT: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point){{$}}

