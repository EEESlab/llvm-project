// Driver for manual testing of OpenMP master directive lowering.
//
// Usage (emit CIR):
//   clang -cc1 -fopenmp -emit-cir -fclangir master-driver.c -o master-driver.cir
//
// Usage (emit LLVM IR):
//   clang -cc1 -fopenmp -emit-llvm -fclangir master-driver.c -o master-driver.ll
//
// Usage (compile & run):
//   clang -fopenmp master-driver.c -o master-driver && OMP_NUM_THREADS=4 ./master-driver

#include <stdio.h>
#include <omp.h>

void use(int x) {
    (void)x;
}

// ============================================================
// Test 1: only the master thread executes the block
// ============================================================
void test_master_only() {
  int executed_by = -1;
#pragma omp parallel
  {
#pragma omp master
    {
      executed_by = omp_get_thread_num();
    }
  }
  printf("master block executed by thread %d (should be 0)\n", executed_by);
}

// ============================================================
// Test 2: master inside parallel — other threads skip the block
// ============================================================
void test_master_skipped_by_others() {
  int count = 0;
#pragma omp parallel
  {
#pragma omp master
    {
#pragma omp atomic
      count++;
    }
  }
  printf("master block executed %d time(s) (should be 1)\n", count);
}
// ============================================================
// Test 3: master with shared variable modification
// ============================================================
void test_master_shared_write() {
  int value = 0;
#pragma omp parallel shared(value)
  {
#pragma omp master
    {
      value = 42;
    }
#pragma omp barrier
    // After the barrier, all threads see value == 42
  }
  printf("shared value after master write = %d (should be 42)\n", value);
}

// ============================================================
// Test 4: nested master inside parallel
// ============================================================
void test_master_nested() {
  int outer_tid = -1, inner_tid = -1;
#pragma omp parallel
  {
#pragma omp master
    {
      outer_tid = omp_get_thread_num();
#pragma omp parallel
      {
#pragma omp master
        {
          inner_tid = omp_get_thread_num();
        }
      }
    }
  }
  printf("outer master tid=%d, inner master tid=%d (both should be 0)\n",
         outer_tid, inner_tid);
}

// ============================================================
// Main
// ============================================================
int main() {
  printf("=== test_master_only ===\n");
  test_master_only();

  printf("\n=== test_master_skipped_by_others ===\n");
  test_master_skipped_by_others();

  printf("\n=== test_master_shared_write ===\n");
  test_master_shared_write();

  printf("\n=== test_master_nested ===\n");
  test_master_nested();

  printf("\nAll master tests completed.\n");
  return 0;
}
