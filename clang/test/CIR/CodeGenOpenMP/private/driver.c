// Driver for manual testing of OpenMP private clause lowering.
//
// Usage (emit CIR):
//   clang -cc1 -fopenmp -emit-cir -fclangir driver.c -o driver.cir
//
// Usage (emit LLVM IR):
//   clang -cc1 -fopenmp -emit-llvm -fclangir driver.c -o driver.ll
//
// Usage (compile to object):
//   clang -fopenmp -fclangir driver.c -c -o driver.o

#include <stdio.h>

// ============================================================
// Parallel private — basic int
// ============================================================
void test_parallel_private_int() {
  int x = 10;
#pragma omp parallel private(x)
  {
    x = 42;
    printf("parallel private int: x = %d\n", x);
  }
  printf("after parallel: x = %d (should be 10)\n", x);
}

// ============================================================
// Parallel private — multiple variables
// ============================================================
void test_parallel_private_multi() {
  int a = 1, b = 2;
#pragma omp parallel private(a, b)
  {
    a = 100;
    b = 200;
    printf("parallel private multi: a=%d b=%d\n", a, b);
  }
  printf("after parallel: a=%d b=%d (should be 1, 2)\n", a, b);
}

// ============================================================
// For private — basic int
// ============================================================
void test_for_private_int() {
  int priv = 0;
#pragma omp parallel
  {
#pragma omp for private(priv)
    for (int i = 0; i < 4; i++) {
      priv = i * 10;
      printf("for private int: i=%d priv=%d\n", i, priv);
    }
  }
  printf("after for: priv = %d (should be 0)\n", priv);
}

// ============================================================
// Nested: parallel private + for private
// ============================================================
void test_nested_private() {
  int x = 1, y = 2;
#pragma omp parallel private(x)
  {
    x = 99;
#pragma omp for private(y)
    for (int i = 0; i < 4; i++) {
      y = i + x;
      printf("nested: i=%d x=%d y=%d\n", i, x, y);
    }
  }
  printf("after nested: x=%d y=%d (should be 1, 2)\n", x, y);
}

// ============================================================
// Different scalar types
// ============================================================
void test_private_float() {
  float f = 3.14f;
#pragma omp parallel private(f)
  {
    f = 2.71f;
    printf("parallel private float: f = %f\n", f);
  }
  printf("after parallel: f = %f (should be ~3.14)\n", f);
}

void test_private_double() {
  double d = 1.41421356;
#pragma omp parallel private(d)
  {
    d = 2.71828;
    printf("parallel private double: d = %f\n", d);
  }
  printf("after parallel: d = %f (should be ~1.414)\n", d);
}

void test_private_pointer() {
  int val = 55;
  int *p = &val;
#pragma omp parallel private(p)
  {
    p = NULL;
    printf("parallel private pointer: p = %p\n", (void *)p);
  }
  printf("after parallel: *p = %d (should be 55)\n", *p);
}

void test_private_short() {
  short s = 7;
#pragma omp parallel private(s)
  {
    s = 123;
    printf("parallel private short: s = %d\n", s);
  }
  printf("after parallel: s = %d (should be 7)\n", s);
}

void test_private_longlong() {
  long long ll = 9999999999LL;
#pragma omp parallel private(ll)
  {
    ll = 42;
    printf("parallel private long long: ll = %lld\n", ll);
  }
  printf("after parallel: ll = %lld (should be 9999999999)\n", ll);
}

void test_private_unsigned() {
  unsigned u = 12345;
#pragma omp parallel private(u)
  {
    u = 0;
    printf("parallel private unsigned: u = %u\n", u);
  }
  printf("after parallel: u = %u (should be 12345)\n", u);
}

// ============================================================
// Main
// ============================================================
int main() {
  printf("=== parallel private int ===\n");
  test_parallel_private_int();

  printf("\n=== parallel private multi ===\n");
  test_parallel_private_multi();

  printf("\n=== for private int ===\n");
  test_for_private_int();

  printf("\n=== nested private ===\n");
  test_nested_private();

  printf("\n=== private float ===\n");
  test_private_float();

  printf("\n=== private double ===\n");
  test_private_double();

  printf("\n=== private pointer ===\n");
  test_private_pointer();

  printf("\n=== private short ===\n");
  test_private_short();

  printf("\n=== private long long ===\n");
  test_private_longlong();

  printf("\n=== private unsigned ===\n");
  test_private_unsigned();

  printf("\nAll tests completed.\n");
  return 0;
}
