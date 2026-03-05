// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Tests for the OpenMP firstprivate clause on omp.parallel and omp.wsloop.
// Firstprivate creates a private copy initialized with the original value.

void use_int(int);
void use_float(float);
void use_ptr(int *);

// --- Test 1: basic firstprivate on parallel ---

// CHECK-LABEL: omp.private {type = firstprivate} @x.privatizer : i32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[INIT_ALLOC:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[INIT_ALLOC]] : !llvm.ptr)
// CHECK:       } copy {
// CHECK:       ^bb0(%[[COPY_ORIG:.*]]: !llvm.ptr, %[[COPY_PRIV:.*]]: !llvm.ptr):
// CHECK:         %[[VAL:.*]] = llvm.load %[[COPY_ORIG]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[VAL]], %[[COPY_PRIV]] : i32, !llvm.ptr
// CHECK:         omp.yield(%[[COPY_PRIV]] : !llvm.ptr)
// CHECK:       }
void test_firstprivate_parallel() {
  int x = 10;
  // CHECK: cir.func{{.*}}@{{.*}}test_firstprivate_parallel

  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // CHECK: omp.parallel private(@x.privatizer %[[X_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s32i>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel firstprivate(x)
  {
    use_int(x);
  }
}

// --- Test 2: firstprivate with float ---

// CHECK-LABEL: omp.private {type = firstprivate} @f.privatizer : f32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[F_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[F_INIT]] : !llvm.ptr)
// CHECK:       } copy {
// CHECK:       ^bb0(%[[F_ORIG:.*]]: !llvm.ptr, %[[F_PRIV:.*]]: !llvm.ptr):
// CHECK:         %[[FVAL:.*]] = llvm.load %[[F_ORIG]] : !llvm.ptr -> f32
// CHECK:         llvm.store %[[FVAL]], %[[F_PRIV]] : f32, !llvm.ptr
// CHECK:         omp.yield(%[[F_PRIV]] : !llvm.ptr)
// CHECK:       }
void test_firstprivate_float() {
  float f = 3.14f;
#pragma omp parallel firstprivate(f)
  {
    use_float(f);
  }
}

// --- Test 3: firstprivate with pointer ---

// CHECK-LABEL: omp.private {type = firstprivate} @p.privatizer : !llvm.ptr init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[P_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[P_INIT]] : !llvm.ptr)
// CHECK:       } copy {
// CHECK:       ^bb0(%[[P_ORIG:.*]]: !llvm.ptr, %[[P_PRIV:.*]]: !llvm.ptr):
// CHECK:         %[[PVAL:.*]] = llvm.load %[[P_ORIG]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.store %[[PVAL]], %[[P_PRIV]] : !llvm.ptr, !llvm.ptr
// CHECK:         omp.yield(%[[P_PRIV]] : !llvm.ptr)
// CHECK:       }
void test_firstprivate_pointer() {
  int *p = 0;
#pragma omp parallel firstprivate(p)
  {
    use_ptr(p);
  }
}

// --- Test 4: mixed private and firstprivate ---

// CHECK-LABEL: omp.private {type = private} @a.privatizer : i32 init {
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       }
// CHECK-LABEL: omp.private {type = firstprivate} @b.privatizer : i32 init {
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       } copy {
// CHECK:         llvm.load
// CHECK:         llvm.store
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       }
void test_mixed_private_firstprivate() {
  int a = 1, b = 2;
  // CHECK: cir.func{{.*}}@{{.*}}test_mixed_private_firstprivate

  // CHECK: omp.parallel private(@a.privatizer %{{.*}} -> %{{.*}}, @b.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr) {
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(a) firstprivate(b)
  {
    use_int(a);
    use_int(b);
  }
}

// --- Test 5: firstprivate on wsloop (omp for) ---

void test_firstprivate_wsloop() {
  int val = 42;
  // CHECK: cir.func{{.*}}@{{.*}}test_firstprivate_wsloop

  // CHECK: omp.wsloop private(@val.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
  // CHECK:   omp.loop_nest
#pragma omp parallel
  {
#pragma omp for firstprivate(val)
    for (int i = 0; i < 10; i++) {
      use_int(val);
    }
  }
}
