# RUN: not llvm-mc -triple=riscv32 --mattr=+xcvealu %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

cv.add.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.add.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.add.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.add.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.add.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.add.d a0, a2, a4, a6
# CHECK-ERROR: :[[@LINE-1]]:22: error: invalid operand for instruction

cv.sub.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.sub.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.sub.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.sub.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.sub.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.and.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.and.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.and.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.and.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.and.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.or.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:9: error: register must be even

cv.or.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:13: error: register must be even

cv.or.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:17: error: register must be even

cv.or.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:17: error: invalid operand for instruction

cv.or.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.xor.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.xor.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.xor.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.xor.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.xor.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.min.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.min.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.min.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.min.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.min.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.minu.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:11: error: register must be even

cv.minu.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:15: error: register must be even

cv.minu.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:19: error: register must be even

cv.minu.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:19: error: invalid operand for instruction

cv.minu.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.max.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.max.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.max.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.max.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.max.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.maxu.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:11: error: register must be even

cv.maxu.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:15: error: register must be even

cv.maxu.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:19: error: register must be even

cv.maxu.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:19: error: invalid operand for instruction

cv.maxu.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.sll.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.sll.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.sll.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.sll.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.sll.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction

cv.srl.d a1, a2, a4
# CHECK-ERROR: :[[@LINE-1]]:10: error: register must be even

cv.srl.d a0, a3, a4
# CHECK-ERROR: :[[@LINE-1]]:14: error: register must be even

cv.srl.d a0, a2, a5
# CHECK-ERROR: :[[@LINE-1]]:18: error: register must be even

cv.srl.d a0, a2, 0
# CHECK-ERROR: :[[@LINE-1]]:18: error: invalid operand for instruction

cv.srl.d a0, a2
# CHECK-ERROR: :[[@LINE-1]]:1: error: too few operands for instruction
