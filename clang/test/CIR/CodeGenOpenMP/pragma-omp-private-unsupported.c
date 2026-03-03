// Tests that the private clause crashes (llvm_unreachable) on non-integer types.
// Each case is a separate RUN line using `not --crash` to verify the crash.
//
// The current implementation in VisitOMPPrivateClause (CIRGenOpenMPClause.cpp)
// only handles cir::IntType. All other element types hit llvm_unreachable.

// RUN: not --crash %clang_cc1 -fopenmp -emit-cir -fclangir \
// RUN:   -DTEST_FLOAT %s -o /dev/null 2>&1
// RUN: not --crash %clang_cc1 -fopenmp -emit-cir -fclangir \
// RUN:   -DTEST_DOUBLE %s -o /dev/null 2>&1
// RUN: not --crash %clang_cc1 -fopenmp -emit-cir -fclangir \
// RUN:   -DTEST_POINTER %s -o /dev/null 2>&1
// RUN: not --crash %clang_cc1 -fopenmp -emit-cir -fclangir \
// RUN:   -DTEST_STRUCT %s -o /dev/null 2>&1

void use_float(float);
void use_double(double);
void use_ptr(int *);

struct S { int a; int b; };
void use_struct(struct S);

#ifdef TEST_FLOAT
void test_private_float() {
  float f = 1.0f;
#pragma omp parallel private(f)
  {
    use_float(f);
  }
}
#endif

#ifdef TEST_DOUBLE
void test_private_double() {
  double d = 2.0;
#pragma omp parallel private(d)
  {
    use_double(d);
  }
}
#endif

#ifdef TEST_POINTER
void test_private_pointer() {
  int *p = 0;
#pragma omp parallel private(p)
  {
    use_ptr(p);
  }
}
#endif

#ifdef TEST_STRUCT
void test_private_struct() {
  struct S s = {1, 2};
#pragma omp parallel private(s)
  {
    use_struct(s);
  }
}
#endif
