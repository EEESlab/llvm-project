// Driver for manual runtime testing of #pragma omp task (not a lit test).
//
// Build (example):
//   clang -fopenmp -fclangir task-driver.c -o task-driver
//   ./task-driver
//
// Expected behavior is described in each test.

#include <omp.h>
#include <stdio.h>

// Test 1: deferred execution — tasks created by one thread (single)
// are executed by the team's threads. The sum check verifies all
// tasks ran exactly once.
int results1[8];
void test_task_basic() {
  for (int i = 0; i < 8; i++)
    results1[i] = 0;
#pragma omp parallel
  {
#pragma omp single
    {
      for (int i = 0; i < 8; i++) {
#pragma omp task firstprivate(i)
        {
          results1[i] = i + 1;
          printf("task %d executed by thread %d\n", i,
                 omp_get_thread_num());
        }
      }
    }
  } // implicit barrier waits for all tasks
  int sum = 0;
  for (int i = 0; i < 8; i++)
    sum += results1[i];
  printf("basic: sum=%d (expected 36)\n", sum);
}

// Test 2: implicit firstprivate — `x` is captured by value at task
// creation; the later write to x must not affect the task.
void test_task_implicit_firstprivate() {
  int x = 100;
  int seen = -1;
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task shared(seen)
      {
        seen = x; // x is implicitly firstprivate (captured = 100)
      }
      x = 999; // must not be visible inside the task
#pragma omp taskwait
    }
  }
  printf("implicit firstprivate: seen=%d (expected 100)\n", seen);
}

// Test 3: if(0) — undeferred task, executed immediately by the
// encountering thread.
void test_task_if_zero() {
  int creator = -1, executor = -2;
#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      creator = omp_get_thread_num();
#pragma omp task if(0) shared(executor)
      {
        executor = omp_get_thread_num();
      }
    }
  }
  printf("if(0): creator=%d executor=%d (expected equal)\n", creator,
         executor);
}

// Test 4: taskwait — child tasks complete before the line after
// taskwait runs.
void test_taskwait() {
  int a = 0, b = 0;
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task shared(a)
      a = 1;
#pragma omp task shared(b)
      b = 2;
#pragma omp taskwait
      printf("taskwait: a=%d b=%d (expected 1 2)\n", a, b);
    }
  }
}

// Test 5: fibonacci with final — classic recursive task pattern;
// final() cuts task creation overhead at deep recursion levels.
int fib(int n) {
  if (n < 2)
    return n;
  int l, r;
#pragma omp task shared(l) final(n < 10)
  l = fib(n - 1);
#pragma omp task shared(r) final(n < 10)
  r = fib(n - 2);
#pragma omp taskwait
  return l + r;
}
void test_task_fib() {
  int result = 0;
#pragma omp parallel
  {
#pragma omp single
    result = fib(15);
  }
  printf("fib(15)=%d (expected 610)\n", result);
}

int main() {
  printf("=== task basic (deferred execution) ===\n");
  test_task_basic();

  printf("\n=== task implicit firstprivate ===\n");
  test_task_implicit_firstprivate();

  printf("\n=== task if(0) (undeferred) ===\n");
  test_task_if_zero();

  printf("\n=== taskwait ===\n");
  test_taskwait();

  printf("\n=== task fib with final ===\n");
  test_task_fib();

  return 0;
}
