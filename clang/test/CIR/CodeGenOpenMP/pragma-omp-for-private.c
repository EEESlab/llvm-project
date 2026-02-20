// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void use(int);

// Test private clause on omp for: the omp.private op is emitted at module
// level, and wsloop references it via private_vars/private_syms.
void emit_for_private() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_private
  int x = 42;
#pragma omp parallel
  {
#pragma omp for private(x)
    for (int i = 0; i < 10; i++) {
      use(x);
    }
  }

  // module-level privatizer (emitted before the function)
  // CHECK: omp.private {type = private} @x.privatizer : i32

  // CHECK: omp.parallel {

  // x alloca (outer, referenced by private_vars operand)
  // CHECK: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]

  // loop bounds
  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // induction variable alloca
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]

  // conversion to std integer for omp.loop_nest
  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[C0_CIR]] : !s32i to i32
  // CHECK: %[[C10:.*]] = builtin.unrealized_conversion_cast %[[C10_CIR]] : !s32i to i32
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[C1_CIR]] : !s32i to i32

  // cast x alloca to !llvm.ptr for the private_vars operand
  // CHECK: %[[X_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[X_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // wsloop with private clause; the wsloop block has one block arg (%[[PRIV_ARG]])
  // CHECK: omp.wsloop private(@x.privatizer %[[X_LLVMPTR]] -> %[[PRIV_ARG:.*]] : !llvm.ptr) {

  // loop_nest is the single op inside the wsloop block
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {

  // Inside the loop_nest body: cast block arg back to CIR pointer for remapping
  // CHECK: %[[PRIV_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_ARG]] : !llvm.ptr to !cir.ptr<!s32i>

  // store induction variable
  // CHECK: %[[IV_CIR:.*]] = builtin.unrealized_conversion_cast %[[IV]] : i32 to !s32i
  // CHECK: cir.store %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // use(x) loads from the private copy
  // CHECK: %[[X_VAL:.*]] = cir.load %[[PRIV_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use(%[[X_VAL]])

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
}

// Test private clause with multiple variables
void emit_for_private_multi() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_private_multi
  int a = 1, b = 2;
#pragma omp parallel
  {
#pragma omp for private(a, b)
    for (int i = 0; i < 10; i++) {
      use(a);
      use(b);
    }
  }

  // CHECK: omp.private {type = private} @a.privatizer : i32
  // CHECK: omp.private {type = private} @b.privatizer : i32

  // CHECK: omp.parallel {

  // CHECK: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
  // CHECK: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]

  // cast to !llvm.ptr for private_vars operands
  // CHECK: %[[A_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[A_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr
  // CHECK: %[[B_LLVMPTR:.*]] = builtin.unrealized_conversion_cast %[[B_ALLOCA]] : !cir.ptr<!s32i> to !llvm.ptr

  // wsloop with two private vars; block has two block args
  // CHECK: omp.wsloop private(@a.privatizer %[[A_LLVMPTR]] -> %[[PRIV_A:.*]], @b.privatizer %[[B_LLVMPTR]] -> %[[PRIV_B:.*]] : !llvm.ptr, !llvm.ptr) {

  // loop_nest is the single op in the wsloop block
  // CHECK-NEXT: omp.loop_nest

  // casts from block args back to CIR pointers, inside loop_nest body
  // CHECK: %[[A_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_A]] : !llvm.ptr to !cir.ptr<!s32i>
  // CHECK: %[[B_CIR:.*]] = builtin.unrealized_conversion_cast %[[PRIV_B]] : !llvm.ptr to !cir.ptr<!s32i>

  // use(a) and use(b) load from private copies
  // CHECK: cir.load %[[A_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use
  // CHECK: cir.load %[[B_CIR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}use

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}
