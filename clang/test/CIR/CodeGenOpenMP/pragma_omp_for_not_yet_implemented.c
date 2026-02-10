// RUN: not %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void test_schedule() {
  int i;

#pragma omp parallel
  {2
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : schedule}}
#pragma omp for schedule(static)
    for (i = 0; i < 10; ++i) {
    }
  }
}

void test_ordered() {
  int i;

#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : ordered}}
#pragma omp for ordered
    for (i = 0; i < 10; ++i) {
    }
  }
}

void test_nowait() {
  int i;

#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : nowait}}
#pragma omp for nowait
    for (i = 0; i < 10; ++i) {
    }
  }
}

void test_collapse() {
  int i, j;

#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : collapse}}
#pragma omp for collapse(2)
    for (i = 0; i < 10; ++i) {
        for(j=0; j< 20; j++) {}
    }
  }
}


void test_private() {
  int i, x;
#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : private}}
#pragma omp for private(x)
    for (i = 0; i < 10; ++i) {
      x = i;
    }
  }
}


void test_firstprivate() {
  int i, x = 100;
#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : firstprivate}}
#pragma omp for firstprivate(x)
    for (i = 0; i < 10; ++i) {
      int y = x + i;
    }
  }
}

void test_lastprivate() {
  int i, x;
#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : lastprivate}}
#pragma omp for lastprivate(x)
    for (i = 0; i < 10; ++i) {
      x = i;
    }
  }
}

void test_reduction() {
  int i, sum = 0;
#pragma omp parallel
  {
    // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMPClause : reduction}}
#pragma omp for reduction(+:sum)
    for (i = 0; i < 10; ++i) {
      sum += i;
    }
  }
}


void emit_nested_for_in_omp_for() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 4; i++) {
      // expected-error@+1{{ClangIR code gen Not Yet Implemented: inc/dec OpenMP}}
      for (int k = 0; k < 2; k++) {
      }
    }
  }
}