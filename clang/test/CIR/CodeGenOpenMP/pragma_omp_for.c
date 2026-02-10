// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void before(int);
void during(int);
void after(int);

void emit_simple_for() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_simple_for
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
        during(j);
    }
  }
  // CHECK: omp.parallel {

  // constants
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[C10:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {

  // during(j)
  // CHECK: cir.load {{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
  after(j);
  // CHECK: cir.call @{{.*}}after
}

void emit_for_with_vars() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_with_vars
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
    int lb = 1;
    long ub = 10;
    short step = 1;
#pragma omp for
    for (int i = 0; i < ub; i=i+step) {
        during(j);
    }
  }

  // CHECK: omp.parallel {

  // allocas
  // CHECK: %[[LB:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["lb", init]
  // CHECK: %[[UB:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["ub", init]
  // CHECK: %[[STEP:.*]] = cir.alloca !s16i, !cir.ptr<!s16i>, ["step", init]

  // stores
  // CHECK: cir.store {{.*}}, %[[LB]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.store {{.*}}, %[[UB]] : !s64i, !cir.ptr<!s64i>
  // CHECK: cir.store {{.*}}, %[[STEP]] : !s16i, !cir.ptr<!s16i>

  // bounds + conversions
  // CHECK: %[[LB0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[UBLOAD:.*]] = cir.load {{.*}} %[[UB]] : !cir.ptr<!s64i>, !s64i
  // CHECK: %[[UBCAST:.*]] = builtin.unrealized_conversion_cast %[[UBLOAD]] : !s64i to i64
  // CHECK: %[[UBTRUNC:.*]] = llvm.trunc %[[UBCAST]] : i64 to i32

  // CHECK: %[[STEPLOAD:.*]] = cir.load {{.*}} %[[STEP]] : !cir.ptr<!s16i>, !s16i
  // CHECK: %[[STEPCONV:.*]] = cir.cast integral %[[STEPLOAD]] : !s16i -> !s32i
  // CHECK: %[[STEPCONV2:.*]] = builtin.unrealized_conversion_cast %[[STEPCONV]] : !s32i to i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[LB0]]) to (%[[UBTRUNC]]) step (%[[STEPCONV2]]) {

  // during(j)
  // CHECK: cir.load {{.*}} %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }

  after(j);
  // CHECK: cir.call @{{.*}}after
}
