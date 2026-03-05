// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Comprehensive tests for the OpenMP private clause on omp.parallel and
// omp.wsloop, covering write-to-private, post-region value preservation,
// different integer widths, unsigned types, and nested directives.

void use_int(int);
void use_short(short);
void use_ll(long long);
void use_uint(unsigned);

// --- Test 1: write to private var inside parallel ---
// Verify that a store inside the parallel region targets the private copy.

// CHECK-LABEL: omp.private {type = private} @x.privatizer : i32 init {
// CHECK:       ^bb0(%{{.*}}: !llvm.ptr, %[[X_INIT:.*]]: !llvm.ptr):
// CHECK:         omp.yield(%[[X_INIT]] : !llvm.ptr)
// CHECK:       }
void test_write_private_parallel() {
  int x = 0;
  // CHECK: cir.func{{.*}}@{{.*}}test_write_private_parallel

  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // CHECK: omp.parallel private(@x.privatizer %[[X_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s32i>

  // Store 42 into the private copy
  // CHECK: %[[C42:.*]] = cir.const #cir.int<42> : !s32i
  // CHECK: cir.store %[[C42]], %[[PRIV_CIR]] : !s32i, !cir.ptr<!s32i>

  // use_int(x) loads from the private copy
  // CHECK: %[[XVAL:.*]] = cir.load %[[PRIV_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use_int(%[[XVAL]])

  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(x)
  {
    x = 42;
    use_int(x);
  }
}

// --- Test 2: original value accessible after the region ---
// Verify that the original alloca is used after the parallel region ends.

void test_post_region_value() {
  int x = 10;
  // CHECK: cir.func{{.*}}@{{.*}}test_post_region_value

  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: omp.parallel private(@x.privatizer %{{.*}} -> %{{.*}} : !llvm.ptr) {
  // CHECK: omp.terminator
  // CHECK: }

  // After the region, use_int(x) loads from the ORIGINAL alloca
  // CHECK: %[[ORIG_VAL:.*]] = cir.load %[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use_int(%[[ORIG_VAL]])
#pragma omp parallel private(x)
  {
    x = 99;
    use_int(x);
  }
  use_int(x);
}

// --- Test 3: short (16-bit integer) ---

// CHECK: omp.private {type = private} @s.privatizer : i16 init {
// CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
// CHECK: }
void test_private_short() {
  short s = 1;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_short

  // CHECK: %[[S_ALLOCA:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["s", init]
  // CHECK: %[[S_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[S_ALLOCA]] : !cir.ptr<!s16i> to !llvm.ptr

  // CHECK: omp.parallel private(@s.privatizer %[[S_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s16i>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!s16i>, !s16i
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(s)
  {
    use_short(s);
  }
}

// --- Test 4: long long (64-bit integer) ---

// CHECK: omp.private {type = private} @ll.privatizer : i64 init {
// CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
// CHECK: }
void test_private_longlong() {
  long long ll = 100;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_longlong

  // CHECK: %[[LL_ALLOCA:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["ll", init]
  // CHECK: %[[LL_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[LL_ALLOCA]] : !cir.ptr<!s64i> to !llvm.ptr

  // CHECK: omp.parallel private(@ll.privatizer %[[LL_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s64i>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!s64i>, !s64i
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(ll)
  {
    use_ll(ll);
  }
}

// --- Test 5: unsigned int ---

// CHECK: omp.private {type = private} @u.privatizer : i32 init {
// CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
// CHECK: }
void test_private_unsigned() {
  unsigned u = 5;
  // CHECK: cir.func{{.*}}@{{.*}}test_private_unsigned

  // CHECK: %[[U_ALLOCA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["u", init]
  // CHECK: %[[U_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[U_ALLOCA]] : !cir.ptr<!u32i> to !llvm.ptr

  // CHECK: omp.parallel private(@u.privatizer %[[U_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!u32i>
  // CHECK: cir.load %[[PRIV_CIR]] : !cir.ptr<!u32i>, !u32i
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(u)
  {
    use_uint(u);
  }
}

// --- Test 6: nested parallel private + for private ---
// private(x) on the parallel, private(y) on the inner for.

// CHECK: omp.private {type = private} @y.privatizer : i32 init {
// CHECK:   omp.yield(%{{.*}} : !llvm.ptr)
// CHECK: }
void test_nested_private() {
  int x = 1, y = 2;
  // CHECK: cir.func{{.*}}@{{.*}}test_nested_private

  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: %[[Y_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]

  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr
  // CHECK: omp.parallel private(@x.privatizer %[[X_LLVMPTR]] -> %[[PX:.*]] : !llvm.ptr) {
  // CHECK: %[[PX_CIR:.*]] = builtin.unrealized_conversion_cast %[[PX]] : !llvm.ptr to !cir.ptr<!s32i>

  // y's alloca inside the parallel body (remapped)
  // CHECK: %[[Y_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[Y_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr
  // CHECK: omp.wsloop private(@y.privatizer %[[Y_LLVMPTR]] -> %[[PY:.*]] : !llvm.ptr) {
  // CHECK: omp.loop_nest
  // CHECK: %[[PY_CIR:.*]] = builtin.unrealized_conversion_cast %[[PY]] : !llvm.ptr to !cir.ptr<!s32i>

  // use_int(x) loads from parallel's private x
  // CHECK: cir.load %[[PX_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use_int
  // use_int(y) loads from wsloop's private y
  // CHECK: cir.load %[[PY_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use_int

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
#pragma omp parallel private(x)
  {
#pragma omp for private(y)
    for (int i = 0; i < 10; i++) {
      use_int(x);
      use_int(y);
    }
  }
}
