// Driver for manual testing of OpenMP single directive lowering.
//
// Usage (emit CIR):
//   clang -cc1 -fopenmp -emit-cir -fclangir single-driver.c -o single-driver.cir
//
// Usage (emit LLVM IR):
//   clang -cc1 -fopenmp -emit-llvm -fclangir single-driver.c -o single-driver.ll
//
// Usage (compile & run):
//   clang -fopenmp single-driver.c -o single-driver && OMP_NUM_THREADS=4 ./single-driver

#include <stdio.h>
#include <omp.h>

void use(int x) {
    (void)x;
}

// ============================================================
// Test 1: only one thread executes the single block
// ============================================================
void test_single_one_execution() {
  int count = 0;
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp atomic
      count++;
    }
  }
  printf("single block executed %d time(s) (should be 1)\n", count);
}

// ============================================================
// Test 2: single with nowait — no barrier after the block
// ============================================================
void test_single_nowait_execution() {
  int executed_by = -1;
#pragma omp parallel
  {
#pragma omp single nowait
    {
      executed_by = omp_get_thread_num();
    }
    // Other threads don't wait here thanks to nowait
  }
  printf("single nowait block executed by thread %d (should be >= 0)\n",
         executed_by);
}

// NOTE: Tests for single+private and single+firstprivate are omitted here
// because the upstream MLIR OpenMP -> LLVM IR translation does not yet support
// privatization on omp.single (checkImplementationStatus rejects it).
// CIR emission is correct — see pragma-omp-single.c for CIR-level tests.

// ============================================================
// Test 3: multiple single blocks — each executed by one thread
// ============================================================
void test_multiple_singles() {
  int count1 = 0, count2 = 0;
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp atomic
      count1++;
    }
#pragma omp single
    {
#pragma omp atomic
      count2++;
    }
  }
  printf("first single executed %d time(s) (should be 1)\n", count1);
  printf("second single executed %d time(s) (should be 1)\n", count2);
}


// ============================================================
// Main
// ============================================================
int main() {
  printf("=== test_single_one_execution ===\n");
  test_single_one_execution();

  printf("\n=== test_single_nowait ===\n");
  test_single_nowait_execution();

  printf("\n=== test_multiple_singles ===\n");
  test_multiple_singles();

  printf("\nAll single tests completed.\n");
  return 0;
}
