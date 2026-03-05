// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Verify that omp.private ops are emitted at module level and that
// omp.parallel references them via the delayed privatization mechanism.

void use(int);

// CHECK-LABEL: omp.private {type = private} @x.privatizer : i32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[PRIV_ALLOC:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[PRIV_ALLOC]] : !llvm.ptr)
// CHECK:       }
void test_parallel_private() {
  int x = 0;
  // CHECK: cir.func{{.*}}@{{.*}}test_parallel_private

  // x alloca
  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

  // cast x to !llvm.ptr for private_vars (emitted BEFORE parallelOp)
  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // parallel op with private clause
  // CHECK: omp.parallel private(@x.privatizer %[[X_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {

  // inside the body: block arg is cast back to CIR pointer
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s32i>

  // use(x) loads from private copy
  // CHECK: %[[X_VAL:.*]] = cir.load %[[PRIV_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use(%[[X_VAL]])

  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(x)
  {
    use(x);
  }
}

// Two private vars
// CHECK-LABEL: omp.private {type = private} @a.privatizer : i32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[A_PRIV_ALLOC:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[A_PRIV_ALLOC]] : !llvm.ptr)
// CHECK:       }
// CHECK-LABEL: omp.private {type = private} @b.privatizer : i32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[B_PRIV_ALLOC:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[B_PRIV_ALLOC]] : !llvm.ptr)
// CHECK:       }
void test_parallel_private_multi() {
  int a = 1, b = 2;
  // CHECK: cir.func{{.*}}@{{.*}}test_parallel_private_multi

  // CHECK: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
  // CHECK: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]

  // CHECK: %[[A_PTR:.*]] = builtin.unrealized_conversion_cast %[[A_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr
  // CHECK: %[[B_PTR:.*]] = builtin.unrealized_conversion_cast %[[B_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // CHECK: omp.parallel private(@a.privatizer %[[A_PTR]] -> %[[PA:.*]], @b.privatizer %[[B_PTR]] -> %[[PB:.*]] : !llvm.ptr, !llvm.ptr) {

  // CHECK: %[[PA_CIR:.*]] = builtin.unrealized_conversion_cast %[[PA]] : !llvm.ptr to !cir.ptr<!s32i>
  // CHECK: %[[PB_CIR:.*]] = builtin.unrealized_conversion_cast %[[PB]] : !llvm.ptr to !cir.ptr<!s32i>

  // CHECK: cir.load %[[PA_CIR]]
  // CHECK: cir.call @{{.*}}use
  // CHECK: cir.load %[[PB_CIR]]
  // CHECK: cir.call @{{.*}}use

  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(a, b)
  {
    use(a);
    use(b);
  }
}
