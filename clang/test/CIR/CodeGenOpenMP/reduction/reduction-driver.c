// Driver for manual testing of OpenMP reduction clause lowering.
//
// Usage (emit CIR):
//   clang -cc1 -fopenmp -emit-cir -fclangir reduction-driver.c -o reduction-driver.cir
//
// Usage (emit LLVM IR):
//   clang -cc1 -fopenmp -emit-llvm -fclangir reduction-driver.c -o reduction-driver.ll
//
// Usage (compile & run):
//   clang -fopenmp reduction-driver.c -o reduction-driver && OMP_NUM_THREADS=4 ./reduction-driver

#include <stdio.h>
#include <omp.h>

void use(int x) {
    (void)x;
}

// ============================================================
// Test 1: reduction(+:sum) — integer addition
// ============================================================
void test_add_int() {
  int sum = 0;
#pragma omp parallel
  {
#pragma omp for reduction(+:sum)
    for (int i = 1; i <= 100; i++) {
      sum += i;
    }
  }
  printf("add int: sum=%d (expected 5050)\n", sum);
}

// ============================================================
// Test 2: reduction(*:prod) — integer multiplication
// ============================================================
void test_mul_int() {
  int prod = 1;
#pragma omp parallel
  {
#pragma omp for reduction(*:prod)
    for (int i = 1; i <= 10; i++) {
      prod *= i;
    }
  }
  printf("mul int: prod=%d (expected 3628800)\n", prod);
}

// ============================================================
// Test 3: reduction(+:fsum) — float addition
// ============================================================
void test_add_float() {
  float fsum = 0.0f;
#pragma omp parallel
  {
#pragma omp for reduction(+:fsum)
    for (int i = 1; i <= 100; i++) {
      fsum += (float)i;
    }
  }
  printf("add float: fsum=%.1f (expected 5050.0)\n", fsum);
}

// ============================================================
// Test 4: reduction(|:flags) — bitwise OR
// ============================================================
void test_bor() {
  int flags = 0;
#pragma omp parallel
  {
#pragma omp for reduction(|:flags)
    for (int i = 0; i < 8; i++) {
      flags |= (1 << i);
    }
  }
  printf("bor: flags=0x%x (expected 0xff)\n", flags);
}

// ============================================================
// Test 5: reduction on parallel (not for)
// ============================================================
void test_parallel_reduction() {
  int total = 0;
#pragma omp parallel reduction(+:total)
  {
    total += 1;
  }
  printf("parallel reduction: total=%d (expected %d)\n",
         total, omp_get_max_threads());
}

// ============================================================
// Main
// ============================================================
int main() {
  printf("=== reduction(+:int) ===\n");
  test_add_int();

  printf("\n=== reduction(*:int) ===\n");
  test_mul_int();

  printf("\n=== reduction(+:float) ===\n");
  test_add_float();

  printf("\n=== reduction(|:int) ===\n");
  test_bor();

  printf("\n=== parallel reduction(+:int) ===\n");
  test_parallel_reduction();

  printf("\nAll reduction tests completed.\n");
  return 0;
}
