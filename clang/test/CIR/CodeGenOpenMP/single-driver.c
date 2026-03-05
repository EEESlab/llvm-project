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
void test_single_nowait() {
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

// ============================================================
// Test 3: single with private
// ============================================================
void test_single_private() {
  int x = 42;
#pragma omp parallel
  {
#pragma omp single private(x)
    {
      // x is private (uninitialized), assign and print
      x = 99;
      printf("single private: x = %d (should be 99)\n", x);
    }
  }
  printf("after single private: x = %d (should be 42)\n", x);
}

// ============================================================
// Test 4: single with firstprivate
// ============================================================
void test_single_firstprivate() {
  int val = 100;
#pragma omp parallel
  {
#pragma omp single firstprivate(val)
    {
      printf("single firstprivate: val = %d (should be 100)\n", val);
      val = 0;
      printf("single firstprivate after modify: val = %d (should be 0)\n", val);
    }
  }
  printf("after single firstprivate: val = %d (should be 100)\n", val);
}

// ============================================================
// Test 5: multiple single blocks — each executed by one thread
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
// Test 6: single inside a for loop
// ============================================================
void test_single_inside_for() {
  int single_count = 0;
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 8; i++) {
#pragma omp single
      {
#pragma omp atomic
        single_count++;
      }
    }
  }
  printf("single inside for executed %d time(s) (should be >= 1)\n",
         single_count);
}

// ============================================================
// Main
// ============================================================
int main() {
  printf("=== test_single_one_execution ===\n");
  test_single_one_execution();

  printf("\n=== test_single_nowait ===\n");
  test_single_nowait();

  printf("\n=== test_single_private ===\n");
  test_single_private();

  printf("\n=== test_single_firstprivate ===\n");
  test_single_firstprivate();

  printf("\n=== test_multiple_singles ===\n");
  test_multiple_singles();

  printf("\n=== test_single_inside_for ===\n");
  test_single_inside_for();

  printf("\nAll single tests completed.\n");
  return 0;
}
