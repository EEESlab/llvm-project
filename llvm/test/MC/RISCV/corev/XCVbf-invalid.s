# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvbf,+xcvbfv %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.fadd.bf16 t0, t1
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.fadd.bf16 t0, t1, 0
# CHECK-ERROR: :[[@LINE-1]]:22: error: invalid operand for instruction

cv.fadd.bf16 t0, t1, t2, rne
# CHECK-ERROR: :[[@LINE-1]]:26: error: invalid operand for instruction

cv.fmadd.bf16 t0, t1, t2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.fsqrt.bf16 t0, t1, t2
# CHECK-ERROR: :[[@LINE-1]]:23: error: invalid operand for instruction

cv.vfadd.bf16 t0, t1, 5
# CHECK-ERROR: :[[@LINE-1]]:23: error: invalid operand for instruction

cv.vfmac.r.bf16 t0, t1
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.vfcvt.h.bf16 t0, t1
# CHECK-ERROR: :[[@LINE-1]]:1: error: instruction requires the following: 'XCVhv' (CORE-V Packed Half-Precision Floating-Point)

cv.fcvt.h.bf16 t0, t1
# CHECK-ERROR: :[[@LINE-1]]:1: error: instruction requires the following: 'Zhinx' (Half Float in Integer)
