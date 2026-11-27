# RUN: llvm-mc -triple=riscv32 --mattr=+xcvealu -M no-aliases -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvealu < %s \
# RUN:     | llvm-objdump --mattr=+xcvealu --no-print-imm-hex -M no-aliases -d -r - \
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

cv.maxu.d a0, a2, a4
# CHECK-INSTR: cv.maxu.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x75,0xe6,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.maxu.d sp, tp, t1
# CHECK-INSTR: cv.maxu.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x71,0x62,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.maxu.d t5, zero, s0
# CHECK-INSTR: cv.maxu.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x7f,0x80,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.maxu.d t3, t3, t3
# CHECK-INSTR: cv.maxu.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x7e,0xce,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sll.d a0, a2, a4
# CHECK-INSTR: cv.sll.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x15,0xe6,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sll.d sp, tp, t1
# CHECK-INSTR: cv.sll.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x11,0x62,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sll.d t5, zero, s0
# CHECK-INSTR: cv.sll.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x1f,0x80,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sll.d t3, t3, t3
# CHECK-INSTR: cv.sll.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x1e,0xce,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srl.d a0, a2, a4
# CHECK-INSTR: cv.srl.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x55,0xe6,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srl.d sp, tp, t1
# CHECK-INSTR: cv.srl.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x51,0x62,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srl.d t5, zero, s0
# CHECK-INSTR: cv.srl.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x5f,0x80,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srl.d t3, t3, t3
# CHECK-INSTR: cv.srl.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x5e,0xce,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sra.d a0, a2, a4
# CHECK-INSTR: cv.sra.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x55,0xe6,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sra.d sp, tp, t1
# CHECK-INSTR: cv.sra.d sp, tp, t1
# CHECK-ENCODING: [0x33,0x51,0x62,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sra.d t5, zero, s0
# CHECK-INSTR: cv.sra.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x5f,0x80,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sra.d t3, t3, t3
# CHECK-INSTR: cv.sra.d t3, t3, t3
# CHECK-ENCODING: [0x33,0x5e,0xce,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.abs.d a0, a2
# CHECK-INSTR: cv.abs.d a0, a2
# CHECK-ENCODING: [0x33,0x05,0x06,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.abs.d sp, tp
# CHECK-INSTR: cv.abs.d sp, tp
# CHECK-ENCODING: [0x33,0x01,0x02,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.abs.d t5, zero
# CHECK-INSTR: cv.abs.d t5, zero
# CHECK-ENCODING: [0x33,0x0f,0x00,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.abs.d t3, t3
# CHECK-INSTR: cv.abs.d t3, t3
# CHECK-ENCODING: [0x33,0x0e,0x0e,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

# The comparisons write a 32-bit register: rd may be odd.
cv.slt.d a0, a2, a4
# CHECK-INSTR: cv.slt.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x25,0xe6,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slt.d ra, tp, t1
# CHECK-INSTR: cv.slt.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x20,0x62,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slt.d t5, zero, s0
# CHECK-INSTR: cv.slt.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x2f,0x80,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slt.d t4, t3, t3
# CHECK-INSTR: cv.slt.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x2e,0xce,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sltu.d a0, a2, a4
# CHECK-INSTR: cv.sltu.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x35,0xe6,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sltu.d ra, tp, t1
# CHECK-INSTR: cv.sltu.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x30,0x62,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sltu.d t5, zero, s0
# CHECK-INSTR: cv.sltu.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x3f,0x80,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sltu.d t4, t3, t3
# CHECK-INSTR: cv.sltu.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x3e,0xce,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.seq.d a0, a2, a4
# CHECK-INSTR: cv.seq.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x25,0xe6,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.seq.d ra, tp, t1
# CHECK-INSTR: cv.seq.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x20,0x62,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.seq.d t5, zero, s0
# CHECK-INSTR: cv.seq.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x2f,0x80,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.seq.d t4, t3, t3
# CHECK-INSTR: cv.seq.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x2e,0xce,0x37]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sne.d a0, a2, a4
# CHECK-INSTR: cv.sne.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x35,0xe6,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sne.d ra, tp, t1
# CHECK-INSTR: cv.sne.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x30,0x62,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sne.d t5, zero, s0
# CHECK-INSTR: cv.sne.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x3f,0x80,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sne.d t4, t3, t3
# CHECK-INSTR: cv.sne.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x3e,0xce,0x37]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slet.d a0, a2, a4
# CHECK-INSTR: cv.slet.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x25,0xe6,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slet.d ra, tp, t1
# CHECK-INSTR: cv.slet.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x20,0x62,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slet.d t5, zero, s0
# CHECK-INSTR: cv.slet.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x2f,0x80,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slet.d t4, t3, t3
# CHECK-INSTR: cv.slet.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x2e,0xce,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sletu.d a0, a2, a4
# CHECK-INSTR: cv.sletu.d a0, a2, a4
# CHECK-ENCODING: [0x33,0x35,0xe6,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sletu.d ra, tp, t1
# CHECK-INSTR: cv.sletu.d ra, tp, t1
# CHECK-ENCODING: [0xb3,0x30,0x62,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sletu.d t5, zero, s0
# CHECK-INSTR: cv.sletu.d t5, zero, s0
# CHECK-ENCODING: [0x33,0x3f,0x80,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.sletu.d t4, t3, t3
# CHECK-INSTR: cv.sletu.d t4, t3, t3
# CHECK-ENCODING: [0xb3,0x3e,0xce,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.cnt.d a0, a2
# CHECK-INSTR: cv.cnt.d a0, a2
# CHECK-ENCODING: [0x33,0x15,0x06,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.cnt.d ra, tp
# CHECK-INSTR: cv.cnt.d ra, tp
# CHECK-ENCODING: [0xb3,0x10,0x02,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.cnt.d t5, zero
# CHECK-INSTR: cv.cnt.d t5, zero
# CHECK-ENCODING: [0x33,0x1f,0x00,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.cnt.d t4, t3
# CHECK-INSTR: cv.cnt.d t4, t3
# CHECK-ENCODING: [0xb3,0x1e,0x0e,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exths.d a0, a2
# CHECK-INSTR: cv.exths.d a0, a2
# CHECK-ENCODING: [0x33,0x05,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exths.d sp, tp
# CHECK-INSTR: cv.exths.d sp, tp
# CHECK-ENCODING: [0x33,0x01,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exths.d t5, zero
# CHECK-INSTR: cv.exths.d t5, zero
# CHECK-ENCODING: [0x33,0x0f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exths.d t3, ra
# CHECK-INSTR: cv.exths.d t3, ra
# CHECK-ENCODING: [0x33,0x8e,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exthz.d a0, a2
# CHECK-INSTR: cv.exthz.d a0, a2
# CHECK-ENCODING: [0x33,0x15,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exthz.d sp, tp
# CHECK-INSTR: cv.exthz.d sp, tp
# CHECK-ENCODING: [0x33,0x11,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exthz.d t5, zero
# CHECK-INSTR: cv.exthz.d t5, zero
# CHECK-ENCODING: [0x33,0x1f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.exthz.d t3, ra
# CHECK-INSTR: cv.exthz.d t3, ra
# CHECK-ENCODING: [0x33,0x9e,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbs.d a0, a2
# CHECK-INSTR: cv.extbs.d a0, a2
# CHECK-ENCODING: [0x33,0x25,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbs.d sp, tp
# CHECK-INSTR: cv.extbs.d sp, tp
# CHECK-ENCODING: [0x33,0x21,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbs.d t5, zero
# CHECK-INSTR: cv.extbs.d t5, zero
# CHECK-ENCODING: [0x33,0x2f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbs.d t3, ra
# CHECK-INSTR: cv.extbs.d t3, ra
# CHECK-ENCODING: [0x33,0xae,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbz.d a0, a2
# CHECK-INSTR: cv.extbz.d a0, a2
# CHECK-ENCODING: [0x33,0x35,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbz.d sp, tp
# CHECK-INSTR: cv.extbz.d sp, tp
# CHECK-ENCODING: [0x33,0x31,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbz.d t5, zero
# CHECK-INSTR: cv.extbz.d t5, zero
# CHECK-ENCODING: [0x33,0x3f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extbz.d t3, ra
# CHECK-INSTR: cv.extbz.d t3, ra
# CHECK-ENCODING: [0x33,0xbe,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extws.d a0, a2
# CHECK-INSTR: cv.extws.d a0, a2
# CHECK-ENCODING: [0x33,0x45,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extws.d sp, tp
# CHECK-INSTR: cv.extws.d sp, tp
# CHECK-ENCODING: [0x33,0x41,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extws.d t5, zero
# CHECK-INSTR: cv.extws.d t5, zero
# CHECK-ENCODING: [0x33,0x4f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extws.d t3, ra
# CHECK-INSTR: cv.extws.d t3, ra
# CHECK-ENCODING: [0x33,0xce,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extwz.d a0, a2
# CHECK-INSTR: cv.extwz.d a0, a2
# CHECK-ENCODING: [0x33,0x55,0x06,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extwz.d sp, tp
# CHECK-INSTR: cv.extwz.d sp, tp
# CHECK-ENCODING: [0x33,0x51,0x02,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extwz.d t5, zero
# CHECK-INSTR: cv.extwz.d t5, zero
# CHECK-ENCODING: [0x33,0x5f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.extwz.d t3, ra
# CHECK-INSTR: cv.extwz.d t3, ra
# CHECK-ENCODING: [0x33,0xde,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slli.d a0, a2, 5
# CHECK-INSTR: cv.slli.d a0, a2, 5
# CHECK-ENCODING: [0x13,0x15,0x56,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slli.d sp, tp, 5
# CHECK-INSTR: cv.slli.d sp, tp, 5
# CHECK-ENCODING: [0x13,0x11,0x52,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slli.d t5, zero, 0
# CHECK-INSTR: cv.slli.d t5, zero, 0
# CHECK-ENCODING: [0x13,0x1f,0x00,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.slli.d t3, t3, 31
# CHECK-INSTR: cv.slli.d t3, t3, 31
# CHECK-ENCODING: [0x13,0x1e,0xfe,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srli.d a0, a2, 5
# CHECK-INSTR: cv.srli.d a0, a2, 5
# CHECK-ENCODING: [0x13,0x55,0x56,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srli.d sp, tp, 5
# CHECK-INSTR: cv.srli.d sp, tp, 5
# CHECK-ENCODING: [0x13,0x51,0x52,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srli.d t5, zero, 0
# CHECK-INSTR: cv.srli.d t5, zero, 0
# CHECK-ENCODING: [0x13,0x5f,0x00,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srli.d t3, t3, 31
# CHECK-INSTR: cv.srli.d t3, t3, 31
# CHECK-ENCODING: [0x13,0x5e,0xfe,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srai.d a0, a2, 5
# CHECK-INSTR: cv.srai.d a0, a2, 5
# CHECK-ENCODING: [0x13,0x55,0x56,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srai.d sp, tp, 5
# CHECK-INSTR: cv.srai.d sp, tp, 5
# CHECK-ENCODING: [0x13,0x51,0x52,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srai.d t5, zero, 0
# CHECK-INSTR: cv.srai.d t5, zero, 0
# CHECK-ENCODING: [0x13,0x5f,0x00,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.srai.d t3, t3, 31
# CHECK-INSTR: cv.srai.d t3, t3, 31
# CHECK-ENCODING: [0x13,0x5e,0xfe,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.addi.d a0, a2, 5
# CHECK-INSTR: cv.addi.d a0, a2, 5
# CHECK-ENCODING: [0x13,0x15,0x56,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.addi.d sp, tp, 5
# CHECK-INSTR: cv.addi.d sp, tp, 5
# CHECK-ENCODING: [0x13,0x11,0x52,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.addi.d t5, zero, -16
# CHECK-INSTR: cv.addi.d t5, zero, -16
# CHECK-ENCODING: [0x13,0x1f,0x00,0x23]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.addi.d t3, t3, 15
# CHECK-INSTR: cv.addi.d t3, t3, 15
# CHECK-ENCODING: [0x13,0x1e,0xfe,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xori.d a0, a2, 5
# CHECK-INSTR: cv.xori.d a0, a2, 5
# CHECK-ENCODING: [0x1b,0x45,0x56,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xori.d sp, tp, 5
# CHECK-INSTR: cv.xori.d sp, tp, 5
# CHECK-ENCODING: [0x1b,0x41,0x52,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xori.d t5, zero, -16
# CHECK-INSTR: cv.xori.d t5, zero, -16
# CHECK-ENCODING: [0x1b,0x4f,0x00,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.xori.d t3, t3, 15
# CHECK-INSTR: cv.xori.d t3, t3, 15
# CHECK-ENCODING: [0x1b,0x4e,0xfe,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.ori.d a0, a2, 5
# CHECK-INSTR: cv.ori.d a0, a2, 5
# CHECK-ENCODING: [0x1b,0x65,0x56,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.ori.d sp, tp, 5
# CHECK-INSTR: cv.ori.d sp, tp, 5
# CHECK-ENCODING: [0x1b,0x61,0x52,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.ori.d t5, zero, -16
# CHECK-INSTR: cv.ori.d t5, zero, -16
# CHECK-ENCODING: [0x1b,0x6f,0x00,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.ori.d t3, t3, 15
# CHECK-INSTR: cv.ori.d t3, t3, 15
# CHECK-ENCODING: [0x1b,0x6e,0xfe,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.andi.d a0, a2, 5
# CHECK-INSTR: cv.andi.d a0, a2, 5
# CHECK-ENCODING: [0x1b,0x75,0x56,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.andi.d sp, tp, 5
# CHECK-INSTR: cv.andi.d sp, tp, 5
# CHECK-ENCODING: [0x1b,0x71,0x52,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.andi.d t5, zero, -16
# CHECK-INSTR: cv.andi.d t5, zero, -16
# CHECK-ENCODING: [0x1b,0x7f,0x00,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}

cv.andi.d t3, t3, 15
# CHECK-INSTR: cv.andi.d t3, t3, 15
# CHECK-ENCODING: [0x1b,0x7e,0xfe,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVealu' (CORE-V 64-bit ALU Operations){{$}}
