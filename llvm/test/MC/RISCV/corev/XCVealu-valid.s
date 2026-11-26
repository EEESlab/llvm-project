# RUN: llvm-mc -triple=riscv32 --mattr=+xcvealu -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvealu < %s \
# RUN:     | llvm-objdump --mattr=+xcvealu -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

# Each operand is an even/odd register pair, written as its even register.

cv.add.d a0, a2, a4
# CHECK-INSTR: cv.add.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x05,0xe6,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.add.d sp, tp, t1
# CHECK-INSTR: cv.add.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x01,0x62,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.add.d t5, zero, s0
# CHECK-INSTR: cv.add.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x0f,0x80,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.add.d t3, t3, t3
# CHECK-INSTR: cv.add.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x0e,0xce,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sub.d a0, a2, a4
# CHECK-INSTR: cv.sub.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x05,0xe6,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sub.d sp, tp, t1
# CHECK-INSTR: cv.sub.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x01,0x62,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sub.d t5, zero, s0
# CHECK-INSTR: cv.sub.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x0f,0x80,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sub.d t3, t3, t3
# CHECK-INSTR: cv.sub.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x0e,0xce,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.and.d a0, a2, a4
# CHECK-INSTR: cv.and.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x75,0xe6,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.and.d sp, tp, t1
# CHECK-INSTR: cv.and.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x71,0x62,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.and.d t5, zero, s0
# CHECK-INSTR: cv.and.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x7f,0x80,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.and.d t3, t3, t3
# CHECK-INSTR: cv.and.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x7e,0xce,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.or.d a0, a2, a4
# CHECK-INSTR: cv.or.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x65,0xe6,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.or.d sp, tp, t1
# CHECK-INSTR: cv.or.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x61,0x62,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.or.d t5, zero, s0
# CHECK-INSTR: cv.or.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x6f,0x80,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.or.d t3, t3, t3
# CHECK-INSTR: cv.or.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x6e,0xce,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xor.d a0, a2, a4
# CHECK-INSTR: cv.xor.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x45,0xe6,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xor.d sp, tp, t1
# CHECK-INSTR: cv.xor.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x41,0x62,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xor.d t5, zero, s0
# CHECK-INSTR: cv.xor.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x4f,0x80,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xor.d t3, t3, t3
# CHECK-INSTR: cv.xor.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x4e,0xce,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.min.d a0, a2, a4
# CHECK-INSTR: cv.min.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x45,0xe6,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.min.d sp, tp, t1
# CHECK-INSTR: cv.min.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x41,0x62,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.min.d t5, zero, s0
# CHECK-INSTR: cv.min.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x4f,0x80,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.min.d t3, t3, t3
# CHECK-INSTR: cv.min.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x4e,0xce,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.minu.d a0, a2, a4
# CHECK-INSTR: cv.minu.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x55,0xe6,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.minu.d sp, tp, t1
# CHECK-INSTR: cv.minu.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x51,0x62,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.minu.d t5, zero, s0
# CHECK-INSTR: cv.minu.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x5f,0x80,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.minu.d t3, t3, t3
# CHECK-INSTR: cv.minu.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x5e,0xce,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.max.d a0, a2, a4
# CHECK-INSTR: cv.max.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x65,0xe6,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.max.d sp, tp, t1
# CHECK-INSTR: cv.max.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x61,0x62,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.max.d t5, zero, s0
# CHECK-INSTR: cv.max.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x6f,0x80,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.max.d t3, t3, t3
# CHECK-INSTR: cv.max.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x6e,0xce,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}
