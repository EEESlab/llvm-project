// Driver for manual testing of OpenMP schedule clause lowering.
//
// Usage (emit CIR):
//   clang -cc1 -fopenmp -emit-cir -fclangir schedule-driver.c -o schedule-driver.cir
//
// Usage (emit LLVM IR):
//   clang -cc1 -fopenmp -emit-llvm -fclangir schedule-driver.c -o schedule-driver.ll
//
// Usage (compile & run):
//   clang -fopenmp schedule-driver.c -o schedule-driver && OMP_NUM_THREADS=4 ./schedule-driver

#include <stdio.h>
#include <omp.h>

// ============================================================
// Test 1: schedule(static) — even distribution
// ============================================================
void test_static() {
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int i = 0; i < 8; i++) {
      printf("static: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Test 2: schedule(dynamic)
// ============================================================
void test_dynamic() {
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < 8; i++) {
      printf("dynamic: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Test 3: schedule(static, 2) — chunks of 2
// ============================================================
void test_static_chunk() {
#pragma omp parallel
  {
#pragma omp for schedule(static, 2)
    for (int i = 0; i < 8; i++) {
      printf("static,2: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Test 4: schedule(dynamic, 3) — chunks of 3
// ============================================================
void test_dynamic_chunk() {
#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 3)
    for (int i = 0; i < 12; i++) {
      printf("dynamic,3: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Test 5: schedule(guided)
// ============================================================
void test_guided() {
#pragma omp parallel
  {
#pragma omp for schedule(guided)
    for (int i = 0; i < 16; i++) {
      printf("guided: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Test 6: schedule(runtime) — uses OMP_SCHEDULE env var
// ============================================================
void test_runtime() {
#pragma omp parallel
  {
#pragma omp for schedule(runtime)
    for (int i = 0; i < 8; i++) {
      printf("runtime: i=%d tid=%d\n", i, omp_get_thread_num());
    }
  }
}

// ============================================================
// Main
// ============================================================
int main() {
  printf("=== schedule(static) ===\n");
  test_static();

  printf("\n=== schedule(dynamic) ===\n");
  test_dynamic();

  printf("\n=== schedule(static, 2) ===\n");
  test_static_chunk();

  printf("\n=== schedule(dynamic, 3) ===\n");
  test_dynamic_chunk();

  printf("\n=== schedule(guided) ===\n");
  test_guided();

  printf("\n=== schedule(runtime) ===\n");
  test_runtime();

  printf("\nAll schedule tests completed.\n");
  return 0;
}
