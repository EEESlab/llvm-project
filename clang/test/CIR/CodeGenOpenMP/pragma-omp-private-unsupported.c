// Tests that the private clause emits a diagnostic (errorNYI) for types that
// are not yet supported by OMPDataSharingProcessor::convertCIRTypeToStdType.
//
// Currently only struct/record types are unsupported. Float, double, pointer,
// and bool are now handled.

// RUN: not %clang_cc1 -fopenmp -emit-cir -fclangir \
// RUN:   -DTEST_STRUCT %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRUCT

struct S { int a; int b; };
void use_struct(struct S);

#ifdef TEST_STRUCT
void test_private_struct() {
  struct S s = {1, 2};
#pragma omp parallel private(s)
  {
    use_struct(s);
  }
}
// STRUCT: not yet implemented: private clause for unsupported type
#endif
