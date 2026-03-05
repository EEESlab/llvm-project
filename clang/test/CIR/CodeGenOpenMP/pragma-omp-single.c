// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_single
void test_single() {
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.single {
#pragma omp single
    {
      // CHECK: cir.call @{{.*}}use
      use(1);
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_single_nowait
void test_single_nowait() {
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.single nowait {
#pragma omp single nowait
    {
      // CHECK: cir.call @{{.*}}use
      use(2);
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}

// CHECK-LABEL: omp.private {type = private} @x.privatizer : i32 init {
// CHECK:       omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_single_private
void test_single_private() {
  int x = 10;
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.single
    // CHECK-SAME: private(@x.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
#pragma omp single private(x)
    {
      // CHECK: cir.call @{{.*}}use
      use(x);
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}

// CHECK-LABEL: omp.private {type = firstprivate} @y.privatizer : i32 init {
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       } copy {
// CHECK:         llvm.load
// CHECK:         llvm.store
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       }

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_single_firstprivate
void test_single_firstprivate() {
  int y = 20;
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.single
    // CHECK-SAME: private(@y.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
#pragma omp single firstprivate(y)
    {
      use(y);
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}
