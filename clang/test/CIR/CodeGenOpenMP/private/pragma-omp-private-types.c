// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Tests that private clause works with non-integer scalar types:
// float, double, pointer, and _Bool.

void use_float(float);
void use_double(double);
void use_ptr(int *);
void use_bool(_Bool);

// --- float ---

// CHECK-LABEL: omp.private {type = private} @f.privatizer : f32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[F_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[F_INIT]] : !llvm.ptr)
// CHECK:       }
void test_private_float() {
  float f = 1.0f;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_float

  // CHECK: %[[F_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
  // CHECK: %[[F_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[F_ALLOCA]] : !cir.ptr<!cir.float> to !llvm.ptr

  // CHECK: omp.parallel private(@f.privatizer %[[F_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!cir.float>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(f)
  {
    use_float(f);
  }
}

// --- double ---

// CHECK-LABEL: omp.private {type = private} @d.privatizer : f64 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[D_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[D_INIT]] : !llvm.ptr)
// CHECK:       }
void test_private_double() {
  double d = 2.0;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_double

  // CHECK: %[[D_ALLOCA:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["d", init]
  // CHECK: %[[D_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[D_ALLOCA]] : !cir.ptr<!cir.double> to !llvm.ptr

  // CHECK: omp.parallel private(@d.privatizer %[[D_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!cir.double>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!cir.double>, !cir.double
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(d)
  {
    use_double(d);
  }
}

// --- pointer ---

// CHECK-LABEL: omp.private {type = private} @p.privatizer : !llvm.ptr init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[P_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[P_INIT]] : !llvm.ptr)
// CHECK:       }
void test_private_pointer() {
  int *p = 0;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_pointer

  // CHECK: %[[P_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]
  // CHECK: %[[P_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[P_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>> to !llvm.ptr

  // CHECK: omp.parallel private(@p.privatizer %[[P_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(p)
  {
    use_ptr(p);
  }
}

// --- _Bool ---

// CHECK-LABEL: omp.private {type = private} @b.privatizer : i1 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[B_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[B_INIT]] : !llvm.ptr)
// CHECK:       }
void test_private_bool() {
  _Bool b = 1;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_bool

  // CHECK: %[[B_ALLOCA:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
  // CHECK: %[[B_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[B_ALLOCA]] : !cir.ptr<!cir.bool> to !llvm.ptr

  // CHECK: omp.parallel private(@b.privatizer %[[B_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!cir.bool>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!cir.bool>, !cir.bool
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(b)
  {
    use_bool(b);
  }
}
