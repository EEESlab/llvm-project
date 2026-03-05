#include <stdio.h>

// These match the functions in pragma-omp-firstprivate.ll
void test_firstprivate_parallel(void);
void test_firstprivate_float(void);
void test_firstprivate_pointer(void);
void test_mixed_private_firstprivate(void);
void test_firstprivate_wsloop(void);

// Callbacks used by the .ll functions
void use_int(int x)    { printf("  use_int   = %d\n", x); }
void use_float(float f){ printf("  use_float = %f\n", f); }
void use_ptr(int *p)   { printf("  use_ptr   = %p\n", (void*)p); }

int main() {
  printf("=== test_firstprivate_parallel ===\n");
  test_firstprivate_parallel();

  printf("\n=== test_firstprivate_float ===\n");
  test_firstprivate_float();

  printf("\n=== test_firstprivate_pointer ===\n");
  test_firstprivate_pointer();

  printf("\n=== test_mixed_private_firstprivate ===\n");
  test_mixed_private_firstprivate();

  printf("\n=== test_firstprivate_wsloop ===\n");
  test_firstprivate_wsloop();

  printf("\nAll firstprivate tests completed.\n");
  return 0;
}