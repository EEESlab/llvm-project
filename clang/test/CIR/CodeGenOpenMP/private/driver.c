// Driver for manual testing of OpenMP private clause lowering.
//
// This file provides:
//   - helper stubs (use, use_int, use_float, ...) called inside omp regions
//   - a main() that calls the test functions defined in the .ll files
//
// Build example (link against a compiled .ll):
//   clang pragma-omp-private.ll driver.c -fopenmp -L/usr/lib/llvm-18/lib -lomp -o test && ./test

#include <stdio.h>

// ============================================================
// Helper stubs — called from inside OpenMP parallel regions
// in the test .ll files. Keep them simple so the output is
// observable but the logic stays in the test source.
// ============================================================

void use(int x)          { printf("  use(int)       = %d\n", x); }
void use_int(int x)      { printf("  use_int        = %d\n", x); }
void use_short(short x)  { printf("  use_short      = %d\n", (int)x); }
void use_ll(long long x) { printf("  use_ll         = %lld\n", x); }
void use_uint(unsigned x){ printf("  use_uint       = %u\n", x); }
void use_float(float x)  { printf("  use_float      = %f\n", x); }
void use_double(double x){ printf("  use_double     = %f\n", x); }
void use_ptr(void *p)    { printf("  use_ptr        = %p\n", p); }
void use_bool(int b)     { printf("  use_bool       = %d\n", b); }

// ============================================================
// Forward declarations — defined in the linked .ll file(s)
// ============================================================

// pragma-omp-private.ll
extern void test_parallel_private(void);
extern void test_parallel_private_multi(void);

// pragma-omp-private-comprehensive.ll  (add/remove as needed)
extern void test_write_private_parallel(void);
extern void test_post_region_value(void);
extern void test_private_short(void);
extern void test_private_longlong(void);
extern void test_private_unsigned(void);
extern void test_nested_private(void);

// pragma-omp-private-types.ll  (add/remove as needed)
extern void test_private_float(void);
extern void test_private_double(void);
extern void test_private_pointer(void);
extern void test_private_bool(void);

// ============================================================
// Main — call whichever tests are defined in the linked .ll
// Comment out groups that are not being linked.
// ============================================================
int main(void) {
  // --- pragma-omp-private ---
  printf("=== test_parallel_private ===\n");
  test_parallel_private();

  printf("\n=== test_parallel_private_multi ===\n");
  test_parallel_private_multi();

  // --- pragma-omp-private-comprehensive ---
  printf("\n=== test_write_private_parallel ===\n");
  test_write_private_parallel();

  printf("\n=== test_post_region_value ===\n");
  test_post_region_value();

  printf("\n=== test_private_short ===\n");
  test_private_short();

  printf("\n=== test_private_longlong ===\n");
  test_private_longlong();

  printf("\n=== test_private_unsigned ===\n");
  test_private_unsigned();

  printf("\n=== test_nested_private ===\n");
  test_nested_private();

  // --- pragma-omp-private-types ---
  printf("\n=== test_private_float ===\n");
  test_private_float();

  printf("\n=== test_private_double ===\n");
  test_private_double();

  printf("\n=== test_private_pointer ===\n");
  test_private_pointer();

  printf("\n=== test_private_bool ===\n");
  test_private_bool();

  printf("\nAll tests completed.\n");
  return 0;
}