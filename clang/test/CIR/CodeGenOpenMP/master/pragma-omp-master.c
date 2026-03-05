// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_master
void test_master() {
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.master {
#pragma omp master
    {
      // CHECK: cir.call @{{.*}}use
      use(42);
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_master_standalone
void test_master_standalone() {
  // The master directive can appear inside any parallel region.
  // CHECK: omp.parallel
#pragma omp parallel
  {
    // CHECK: omp.master {
#pragma omp master
    {
      int x = 1;
      // CHECK: cir.alloca !s32i
      // CHECK: omp.terminator
    }
    // CHECK: omp.terminator
  }
}
