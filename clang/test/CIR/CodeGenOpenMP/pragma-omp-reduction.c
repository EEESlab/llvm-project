// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// CHECK: omp.declare_reduction @add_sum : i32 init {
// CHECK: ^bb0(%{{.*}}: i32):
// CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:   omp.yield(%[[ZERO]] : i32)
// CHECK: } combiner {
// CHECK: ^bb0(%[[A0:.*]]: i32, %[[A1:.*]]: i32):
// CHECK:   %[[RES:.*]] = arith.addi %[[A0]], %[[A1]] : i32
// CHECK:   omp.yield(%[[RES]] : i32)
// CHECK: }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_add_for
void test_reduction_add_for() {
  int sum = 0;
  // CHECK: omp.wsloop reduction(@add_sum %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel
  {
#pragma omp for reduction(+:sum)
    for (int i = 0; i < 10; i++) {
      sum += i;
    }
  }
}

// CHECK: omp.declare_reduction @multiply_prod : i32 init {
// CHECK: ^bb0(%{{.*}}: i32):
// CHECK:   %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:   omp.yield(%[[ONE]] : i32)
// CHECK: } combiner {
// CHECK: ^bb0(%[[A0:.*]]: i32, %[[A1:.*]]: i32):
// CHECK:   %[[RES:.*]] = arith.muli %[[A0]], %[[A1]] : i32
// CHECK:   omp.yield(%[[RES]] : i32)
// CHECK: }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_mul_for
void test_reduction_mul_for() {
  int prod = 1;
  // CHECK: omp.wsloop reduction(@multiply_prod %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel
  {
#pragma omp for reduction(*:prod)
    for (int i = 1; i < 10; i++) {
      prod *= i;
    }
  }
}

// CHECK: omp.declare_reduction @add_fsum : f32 init {
// CHECK: ^bb0(%{{.*}}: f32):
// CHECK:   %[[FZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   omp.yield(%[[FZERO]] : f32)
// CHECK: } combiner {
// CHECK: ^bb0(%[[A0:.*]]: f32, %[[A1:.*]]: f32):
// CHECK:   %[[RES:.*]] = arith.addf %[[A0]], %[[A1]] : f32
// CHECK:   omp.yield(%[[RES]] : f32)
// CHECK: }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_add_float
void test_reduction_add_float() {
  float fsum = 0.0f;
  // CHECK: omp.wsloop reduction(@add_fsum %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel
  {
#pragma omp for reduction(+:fsum)
    for (int i = 0; i < 10; i++) {
      fsum += (float)i;
    }
  }
}

// CHECK: omp.declare_reduction @band_mask : i32 init {
// CHECK: ^bb0(%{{.*}}: i32):
// CHECK:   %[[ONES:.*]] = arith.constant 1 : i32
// CHECK:   omp.yield(%[[ONES]] : i32)
// CHECK: } combiner {
// CHECK: ^bb0(%[[A0:.*]]: i32, %[[A1:.*]]: i32):
// CHECK:   %[[RES:.*]] = arith.andi %[[A0]], %[[A1]] : i32
// CHECK:   omp.yield(%[[RES]] : i32)
// CHECK: }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_bitwise_and
void test_reduction_bitwise_and() {
  int mask = ~0;
  // CHECK: omp.wsloop reduction(@band_mask %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel
  {
#pragma omp for reduction(&:mask)
    for (int i = 0; i < 10; i++) {
      mask &= i;
    }
  }
}

// CHECK: omp.declare_reduction @bor_flags : i32 init {
// CHECK: ^bb0(%{{.*}}: i32):
// CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:   omp.yield(%[[ZERO]] : i32)
// CHECK: } combiner {
// CHECK: ^bb0(%[[A0:.*]]: i32, %[[A1:.*]]: i32):
// CHECK:   %[[RES:.*]] = arith.ori %[[A0]], %[[A1]] : i32
// CHECK:   omp.yield(%[[RES]] : i32)
// CHECK: }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_bitwise_or
void test_reduction_bitwise_or() {
  int flags = 0;
  // CHECK: omp.wsloop reduction(@bor_flags %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel
  {
#pragma omp for reduction(|:flags)
    for (int i = 0; i < 10; i++) {
      flags |= (1 << i);
    }
  }
}

// Test reduction on parallel (not wsloop).
// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_reduction_parallel
void test_reduction_parallel() {
  int total = 0;
  // CHECK: omp.parallel reduction(@add_total %{{.*}} -> %{{.*}} : !llvm.ptr)
#pragma omp parallel reduction(+:total)
  {
    total += 1;
  }
}
