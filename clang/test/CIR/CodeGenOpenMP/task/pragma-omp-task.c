// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Tests for the OpenMP task directive (omp.task) and its clauses.

void use_int(int);

// --- Test 1: bare task with implicit firstprivate ---
//
// A local variable referenced inside a task region is firstprivate by
// default (OpenMP spec). Sema materializes this as an implicit
// firstprivate clause, so the privatizer must appear even though the
// pragma has no explicit clause.

// CHECK-LABEL: omp.private {type = firstprivate} @x.privatizer : i32 init {
// CHECK:       } copy {
// CHECK:         llvm.load
// CHECK:         llvm.store
// CHECK:       }
void test_task_implicit_firstprivate() {
  int x = 10;
  // CHECK: cir.func{{.*}}@{{.*}}test_task_implicit_firstprivate

  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // CHECK: omp.task private(@x.privatizer %[[X_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s32i>
  // CHECK: cir.load{{.*}} %[[PRIV_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp task
  {
    use_int(x);
  }
}

// --- Test 2: explicit firstprivate ---

void test_task_explicit_firstprivate() {
  int y = 20;
  // CHECK: cir.func{{.*}}@{{.*}}test_task_explicit_firstprivate

  // CHECK: omp.task private(@y.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp task firstprivate(y)
  {
    use_int(y);
  }
}

// --- Test 3: explicit private ---

// CHECK-LABEL: omp.private {type = private} @z.privatizer : i32 init {
// CHECK:         omp.yield(%{{.*}} : !llvm.ptr)
// CHECK:       }
void test_task_private() {
  int z = 30;
  // CHECK: cir.func{{.*}}@{{.*}}test_task_private

  // CHECK: omp.task private(@z.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp task private(z)
  {
    z = 1;
    use_int(z);
  }
}

// --- Test 4: if clause ---
//
// The condition is evaluated as a CIR bool and bridged to i1 with an
// unrealized cast (resolved during lowering, like num_threads).

void test_task_if(int c) {
  // CHECK: cir.func{{.*}}@{{.*}}test_task_if

  // CHECK: %[[C_BOOL:.*]] = cir.cast int_to_bool %{{.*}} : !s32i -> !cir.bool
  // CHECK: %[[C_I1:.*]] = builtin.unrealized_conversion_cast %[[C_BOOL]] : !cir.bool to i1
  // CHECK: omp.task if(%[[C_I1]])
#pragma omp task if(c)
  {
    use_int(0);
  }
}

// --- Test 5: final clause ---

void test_task_final(int depth) {
  // CHECK: cir.func{{.*}}@{{.*}}test_task_final

  // CHECK: %[[F_BOOL:.*]] = cir.cmp(gt
  // CHECK: %[[F_I1:.*]] = builtin.unrealized_conversion_cast %[[F_BOOL]] : !cir.bool to i1
  // CHECK: omp.task final(%[[F_I1]])
#pragma omp task final(depth > 4)
  {
    use_int(0);
  }
}

// --- Test 6: priority clause ---

void test_task_priority(int p) {
  // CHECK: cir.func{{.*}}@{{.*}}test_task_priority

  // CHECK: %[[P_LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[P_I32:.*]] = builtin.unrealized_conversion_cast %[[P_LOAD]] : !s32i to i32
  // CHECK: omp.task {{.*}}priority(%[[P_I32]] : i32)
#pragma omp task priority(p)
  {
    use_int(0);
  }
}

// --- Test 7: untied and mergeable ---

void test_task_untied_mergeable() {
  // CHECK: cir.func{{.*}}@{{.*}}test_task_untied_mergeable

  // The printed clause order follows the dialect definition:
  // mergeable comes before untied.
  // CHECK: omp.task mergeable{{.*}}untied
#pragma omp task untied mergeable
  {
    use_int(0);
  }
}

// --- Test 8: task inside parallel + taskwait ---
//
// A variable shared in the enclosing parallel region stays shared in
// the task (no implicit firstprivate), so no privatizer is expected
// for `n`. The taskwait directive lowers to omp.taskwait.

void test_task_in_parallel(int n) {
  // CHECK: cir.func{{.*}}@{{.*}}test_task_in_parallel

  // CHECK: omp.parallel {
  // CHECK: omp.task {
  // CHECK: omp.terminator
  // CHECK: }
  // CHECK: omp.taskwait
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel
  {
#pragma omp task
    {
      use_int(n);
    }
#pragma omp taskwait
  }
}
