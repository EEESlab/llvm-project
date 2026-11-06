# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvhv %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.vfadd.h t0, t1
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.vfadd.h t0, t1, 0
# CHECK-ERROR: :[[@LINE-1]]:20: error: invalid operand for instruction

cv.vfadd.h t0, t1, t2, t3
# CHECK-ERROR: :[[@LINE-1]]:24: error: invalid operand for instruction

cv.vfmac.r.h t0, t1, 5
# CHECK-ERROR: :[[@LINE-1]]:22: error: invalid operand for instruction

cv.vfsqrt.h t0, t1, t2
# CHECK-ERROR: :[[@LINE-1]]:21: error: invalid operand for instruction

cv.vfcvt.x.h t0
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.vfadd.ah t0, t1, t2
# CHECK-ERROR: :[[@LINE-1]]:1: error: unrecognized instruction mnemonic
