// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// Simple parallel for with constant bounds.
void emit_parallel_for_simple() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_parallel_for_simple
  int j = 5;

#pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    during(j);
  }

  // CHECK: omp.parallel {

  // CIR constants for bounds
  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // induction variable alloca
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer
  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[C0_CIR]] : !s32i to i32
  // CHECK: %[[C10:.*]] = builtin.unrealized_conversion_cast %[[C10_CIR]] : !s32i to i32
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[C1_CIR]] : !s32i to i32

  // wsloop nested inside parallel
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {

  // store IV into alloca
  // CHECK: %[[IV_CIR:.*]] = builtin.unrealized_conversion_cast %[[IV]] : i32 to !s32i
  // CHECK: cir.store %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Parallel for with schedule clause.
void emit_parallel_for_schedule() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_parallel_for_schedule

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < 100; i++) {
    during(i);
  }

  // CHECK: omp.parallel {
  // CHECK: omp.wsloop schedule(dynamic, chunk_size = 4 : i32) {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Parallel for with private clause.
void emit_parallel_for_private() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_parallel_for_private
  int x = 42;

#pragma omp parallel for private(x)
  for (int i = 0; i < 10; i++) {
    during(x);
  }

  // CHECK: omp.parallel
  // CHECK-SAME: private(@{{.*}}x{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>)
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Parallel for with reduction clause.
void emit_parallel_for_reduction() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_parallel_for_reduction
  int sum = 0;

#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < 10; i++) {
    sum += i;
  }

  // CHECK: omp.parallel
  // CHECK-SAME: reduction(@{{.*}}sum{{.*}} %{{.*}} -> %{{.*}} : !llvm.ptr)
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}
