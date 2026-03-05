// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_static
void test_schedule_static() {
  // CHECK: omp.wsloop schedule(static)
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_dynamic
void test_schedule_dynamic() {
  // CHECK: omp.wsloop schedule(dynamic)
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_guided
void test_schedule_guided() {
  // CHECK: omp.wsloop schedule(guided)
#pragma omp parallel
  {
#pragma omp for schedule(guided)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_auto
void test_schedule_auto() {
  // CHECK: omp.wsloop schedule(auto)
#pragma omp parallel
  {
#pragma omp for schedule(auto)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_runtime
void test_schedule_runtime() {
  // CHECK: omp.wsloop schedule(runtime)
#pragma omp parallel
  {
#pragma omp for schedule(runtime)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_static_chunk
void test_schedule_static_chunk() {
  // CHECK: omp.wsloop schedule(static = %{{.*}} : i32)
#pragma omp parallel
  {
#pragma omp for schedule(static, 4)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_dynamic_chunk
void test_schedule_dynamic_chunk() {
  // CHECK: omp.wsloop schedule(dynamic = %{{.*}} : i32)
#pragma omp parallel
  {
#pragma omp for schedule(dynamic, 2)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_monotonic
void test_schedule_monotonic() {
  // CHECK: omp.wsloop schedule(dynamic, monotonic)
#pragma omp parallel
  {
#pragma omp for schedule(monotonic: dynamic)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}

// CHECK-LABEL: cir.func{{.*}}@{{.*}}test_schedule_nonmonotonic
void test_schedule_nonmonotonic() {
  // CHECK: omp.wsloop schedule(dynamic, nonmonotonic)
#pragma omp parallel
  {
#pragma omp for schedule(nonmonotonic: dynamic)
    for (int i = 0; i < 10; i++) {
      use(i);
    }
  }
}
