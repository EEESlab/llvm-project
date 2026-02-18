// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file %t-cir.ll

void use(int);

// Test private clause creates a separate alloca in the outlined function
void emit_for_private() {
  int x = 42;
#pragma omp parallel
  {
#pragma omp for private(x)
    for (int i = 0; i < 10; i++) {
      use(x);
    }
  }
}

// CHECK-LABEL: define dso_local void @emit_for_private()
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @emit_for_private..omp_par, ptr %{{.*}})

// CHECK-LABEL: define internal void @emit_for_private..omp_par(
// The private variable gets its own alloca inside the outlined function
// CHECK: %[[PRIV_X:.*]] = alloca i32, align 4
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp.loop_nest.region:
// The loop body uses the private alloca
// CHECK: %[[X_VAL:.*]] = load i32, ptr %[[PRIV_X]], align 4
// CHECK: call void @use(i32 %[[X_VAL]])
// CHECK: call void @__kmpc_for_static_fini(

// Test private clause with multiple variables
void emit_for_private_multi() {
  int a = 1, b = 2;
#pragma omp parallel
  {
#pragma omp for private(a, b)
    for (int i = 0; i < 10; i++) {
      use(a);
      use(b);
    }
  }
}

// CHECK-LABEL: define dso_local void @emit_for_private_multi()
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @emit_for_private_multi..omp_par, ptr %{{.*}})

// CHECK-LABEL: define internal void @emit_for_private_multi..omp_par(
// Both private variables get their own allocas
// CHECK-DAG: %[[PRIV_A:.*]] = alloca i32, align 4
// CHECK-DAG: %[[PRIV_B:.*]] = alloca i32, align 4
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp.loop_nest.region:
// CHECK: call void @use(i32 %{{.*}})
// CHECK: call void @use(i32 %{{.*}})
// CHECK: call void @__kmpc_for_static_fini(
