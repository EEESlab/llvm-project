// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// Test num_threads with a constant (#pragma omp parallel num_threads(4))
void emit_num_threads_const() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_num_threads_const
#pragma omp parallel num_threads(4)
  {
    during(0);
  }
  // The num_threads value is materialized as a CIR constant and bridged to
  // a standard MLIR integer before being attached as an operand.
  // CHECK: %[[NT_CIR:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK: %[[NT:.*]] = builtin.unrealized_conversion_cast %[[NT_CIR]] : !s32i to i32
  // CHECK: omp.parallel num_threads(%[[NT]] : i32) {
  // CHECK: omp.terminator
  // CHECK: }
}

// Test num_threads with a variable (#pragma omp parallel num_threads(n))
void emit_num_threads_var(int n) {
  // CHECK: cir.func{{.*}}@{{.*}}emit_num_threads_var
#pragma omp parallel num_threads(n)
  {
    during(0);
  }
  // CHECK: %[[N_LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[NV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to i32
  // CHECK: omp.parallel num_threads(%[[NV]] : i32) {
  // CHECK: omp.terminator
  // CHECK: }
}
