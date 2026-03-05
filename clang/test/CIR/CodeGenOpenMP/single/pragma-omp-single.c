// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// NOTE: private/firstprivate on omp.single emit correct CIR, but the upstream
// MLIR OpenMP -> LLVM IR lowering does not yet support privatization on
// omp.single (checkImplementationStatus in OpenMPToLLVMIRTranslation.cpp).
// These tests verify CIR emission only.

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