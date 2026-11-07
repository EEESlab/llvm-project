# RUN: llvm-mc -triple=riscv32 --mattr=+xcvbf,+zhinx -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbf,+zhinx < %s \
# RUN:     | llvm-objdump --mattr=+xcvbf,+zhinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 -mattr=+zhinx %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.fmadd.bf16 a0, a1, a2, a3
# CHECK-INSTR: cv.fmadd.bf16 a0, a1, a2, a3
# CHECK-ENCODING: [0x43,0xd5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmadd.bf16 t6, s0, t0, s11
# CHECK-INSTR: cv.fmadd.bf16 t6, s0, t0, s11
# CHECK-ENCODING: [0xc3,0x5f,0x54,0xdc]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmsub.bf16 a0, a1, a2, a3
# CHECK-INSTR: cv.fmsub.bf16 a0, a1, a2, a3
# CHECK-ENCODING: [0x47,0xd5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmsub.bf16 t6, s0, t0, s11
# CHECK-INSTR: cv.fmsub.bf16 t6, s0, t0, s11
# CHECK-ENCODING: [0xc7,0x5f,0x54,0xdc]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fnmsub.bf16 a0, a1, a2, a3
# CHECK-INSTR: cv.fnmsub.bf16 a0, a1, a2, a3
# CHECK-ENCODING: [0x4b,0xd5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fnmsub.bf16 t6, s0, t0, s11
# CHECK-INSTR: cv.fnmsub.bf16 t6, s0, t0, s11
# CHECK-ENCODING: [0xcb,0x5f,0x54,0xdc]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fnmadd.bf16 a0, a1, a2, a3
# CHECK-INSTR: cv.fnmadd.bf16 a0, a1, a2, a3
# CHECK-ENCODING: [0x4f,0xd5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fnmadd.bf16 t6, s0, t0, s11
# CHECK-INSTR: cv.fnmadd.bf16 t6, s0, t0, s11
# CHECK-ENCODING: [0xcf,0x5f,0x54,0xdc]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fadd.bf16 a0, a1, a2
# CHECK-INSTR: cv.fadd.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fadd.bf16 t6, s0, t0
# CHECK-INSTR: cv.fadd.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsub.bf16 a0, a1, a2
# CHECK-INSTR: cv.fsub.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsub.bf16 t6, s0, t0
# CHECK-INSTR: cv.fsub.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmul.bf16 a0, a1, a2
# CHECK-INSTR: cv.fmul.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmul.bf16 t6, s0, t0
# CHECK-INSTR: cv.fmul.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fdiv.bf16 a0, a1, a2
# CHECK-INSTR: cv.fdiv.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fdiv.bf16 t6, s0, t0
# CHECK-INSTR: cv.fdiv.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsqrt.bf16 a0, a1
# CHECK-INSTR: cv.fsqrt.bf16 a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x05,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsqrt.bf16 t6, s0
# CHECK-INSTR: cv.fsqrt.bf16 t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x04,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnj.bf16 a0, a1, a2
# CHECK-INSTR: cv.fsgnj.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xc5,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnj.bf16 t6, s0, t0
# CHECK-INSTR: cv.fsgnj.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x4f,0x54,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnjn.bf16 a0, a1, a2
# CHECK-INSTR: cv.fsgnjn.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnjn.bf16 t6, s0, t0
# CHECK-INSTR: cv.fsgnjn.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnjx.bf16 a0, a1, a2
# CHECK-INSTR: cv.fsgnjx.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xe5,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fsgnjx.bf16 t6, s0, t0
# CHECK-INSTR: cv.fsgnjx.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x6f,0x54,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmin.bf16 a0, a1, a2
# CHECK-INSTR: cv.fmin.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xc5,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmin.bf16 t6, s0, t0
# CHECK-INSTR: cv.fmin.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x4f,0x54,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmax.bf16 a0, a1, a2
# CHECK-INSTR: cv.fmax.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fmax.bf16 t6, s0, t0
# CHECK-INSTR: cv.fmax.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.feq.bf16 a0, a1, a2
# CHECK-INSTR: cv.feq.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xe5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.feq.bf16 t6, s0, t0
# CHECK-INSTR: cv.feq.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x6f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.flt.bf16 a0, a1, a2
# CHECK-INSTR: cv.flt.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xd5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.flt.bf16 t6, s0, t0
# CHECK-INSTR: cv.flt.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x5f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fle.bf16 a0, a1, a2
# CHECK-INSTR: cv.fle.bf16 a0, a1, a2
# CHECK-ENCODING: [0x53,0xc5,0xc5,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fle.bf16 t6, s0, t0
# CHECK-INSTR: cv.fle.bf16 t6, s0, t0
# CHECK-ENCODING: [0xd3,0x4f,0x54,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.w.bf16 a0, a1
# CHECK-INSTR: cv.fcvt.w.bf16 a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x05,0xc4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.w.bf16 t6, s0
# CHECK-INSTR: cv.fcvt.w.bf16 t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x04,0xc4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.wu.bf16 a0, a1
# CHECK-INSTR: cv.fcvt.wu.bf16 a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x15,0xc4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.wu.bf16 t6, s0
# CHECK-INSTR: cv.fcvt.wu.bf16 t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x14,0xc4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.w a0, a1
# CHECK-INSTR: cv.fcvt.bf16.w a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x05,0xd4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.w t6, s0
# CHECK-INSTR: cv.fcvt.bf16.w t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x04,0xd4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.wu a0, a1
# CHECK-INSTR: cv.fcvt.bf16.wu a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x15,0xd4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.wu t6, s0
# CHECK-INSTR: cv.fcvt.bf16.wu t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x14,0xd4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fclass.bf16 a0, a1
# CHECK-INSTR: cv.fclass.bf16 a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x05,0xe4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fclass.bf16 t6, s0
# CHECK-INSTR: cv.fclass.bf16 t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x04,0xe4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.s.bf16 a0, a1
# CHECK-INSTR: cv.fcvt.s.bf16 a0, a1
# CHECK-ENCODING: [0x53,0x85,0x65,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.s.bf16 t6, s0
# CHECK-INSTR: cv.fcvt.s.bf16 t6, s0
# CHECK-ENCODING: [0xd3,0x0f,0x64,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.s a0, a1
# CHECK-INSTR: cv.fcvt.bf16.s a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x05,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.s t6, s0
# CHECK-INSTR: cv.fcvt.bf16.s t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x04,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, dyn
# CHECK-ENCODING: [0x53,0xf5,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 t6, s0
# CHECK-INSTR: cv.fcvt.h.bf16 t6, s0, dyn
# CHECK-ENCODING: [0xd3,0x7f,0x64,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.h a0, a1
# CHECK-INSTR: cv.fcvt.bf16.h a0, a1
# CHECK-ENCODING: [0x53,0xd5,0x25,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.bf16.h t6, s0
# CHECK-INSTR: cv.fcvt.bf16.h t6, s0
# CHECK-ENCODING: [0xd3,0x5f,0x24,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, dyn
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, dyn
# CHECK-ENCODING: [0x53,0xf5,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, rne
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, rne
# CHECK-ENCODING: [0x53,0x85,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, rtz
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, rtz
# CHECK-ENCODING: [0x53,0x95,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, rdn
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, rdn
# CHECK-ENCODING: [0x53,0xa5,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, rup
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, rup
# CHECK-ENCODING: [0x53,0xb5,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fcvt.h.bf16 a0, a1, rmm
# CHECK-INSTR: cv.fcvt.h.bf16 a0, a1, rmm
# CHECK-ENCODING: [0x53,0xc5,0x65,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fneg.bf16 a0, a1
# CHECK-INSTR: cv.fsgnjn.bf16 a0, a1, a1
# CHECK-ENCODING: [0x53,0xd5,0xb5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fabs.bf16 a0, a1
# CHECK-INSTR: cv.fsgnjx.bf16 a0, a1, a1
# CHECK-ENCODING: [0x53,0xe5,0xb5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fgt.bf16 a0, a1, a2
# CHECK-INSTR: cv.flt.bf16 a0, a2, a1
# CHECK-ENCODING: [0x53,0x55,0xb6,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}

cv.fge.bf16 a0, a1, a2
# CHECK-INSTR: cv.fle.bf16 a0, a2, a1
# CHECK-ENCODING: [0x53,0x45,0xb6,0xa4]
# CHECK-NO-EXT: instruction requires the following: 'XCVbf' (CORE-V Scalar BFloat16 Floating-Point){{$}}
