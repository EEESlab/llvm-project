; ModuleID = '/home/lucap/eeeslab/llvm-project/clang/test/CIR/CodeGenOpenMP/reduction/pragma-omp-reduction.c'
source_filename = "/home/lucap/eeeslab/llvm-project/clang/test/CIR/CodeGenOpenMP/reduction/pragma-omp-reduction.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8
@.gomp_critical_user_.reduction.var = common global [8 x i32] zeroinitializer, align 8

; Function Attrs: noinline
define dso_local void @test_reduction_add_for() #0 {
  %structArg = alloca { ptr, ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  store i32 0, ptr %2, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  %gep_9 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %2, ptr %gep_9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_add_for..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_add_for..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca i32, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  store i32 0, ptr %loadgep_, align 4
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region1
  %omp_orig = load i32, ptr %loadgep_2, align 4
  store i32 0, ptr %2, align 4
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.reduction.init
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 9, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num5, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %6
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num6)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num7, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont3
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont3
  %red.value.0 = load i32, ptr %loadgep_2, align 4
  %red.private.value.0 = load i32, ptr %2, align 4
  %7 = add i32 %red.value.0, %red.private.value.0
  store i32 %7, ptr %loadgep_2, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num7, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont3
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %reduce.finalize
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i32 %omp_loop.iv, %3
  %9 = mul i32 %8, 1
  %10 = add i32 %9, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  store i32 %10, ptr %loadgep_, align 4
  %11 = load i32, ptr %loadgep_, align 4
  %12 = load i32, ptr %2, align 1
  %13 = add nsw i32 %12, %11
  store i32 %13, ptr %2, align 1
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i32 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_reduction_mul_for() #0 {
  %structArg = alloca { ptr, ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  store i32 1, ptr %2, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  %gep_9 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %2, ptr %gep_9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_mul_for..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_mul_for..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca i32, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  store i32 1, ptr %loadgep_, align 4
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region1
  %omp_orig = load i32, ptr %loadgep_2, align 4
  store i32 1, ptr %2, align 4
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.reduction.init
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 8, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num5, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %6
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num6)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num7, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func.1, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont3
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont3
  %red.value.0 = load i32, ptr %loadgep_2, align 4
  %red.private.value.0 = load i32, ptr %2, align 4
  %7 = mul i32 %red.value.0, %red.private.value.0
  store i32 %7, ptr %loadgep_2, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num7, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont3
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %reduce.finalize
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i32 %omp_loop.iv, %3
  %9 = mul i32 %8, 1
  %10 = add i32 %9, 1
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  store i32 %10, ptr %loadgep_, align 4
  %11 = load i32, ptr %loadgep_, align 4
  %12 = load i32, ptr %2, align 1
  %13 = mul nsw i32 %12, %11
  store i32 %13, ptr %2, align 1
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i32 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_reduction_add_float() #0 {
  %structArg = alloca { ptr, ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  %2 = alloca float, i64 1, align 4
  store float 0.000000e+00, ptr %2, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  %gep_9 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %2, ptr %gep_9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_add_float..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_add_float..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca float, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  store i32 0, ptr %loadgep_, align 4
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region1
  %omp_orig = load float, ptr %loadgep_2, align 4
  store float 0.000000e+00, ptr %2, align 4
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.reduction.init
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 9, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num5, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %6
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num6)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num7, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func.2, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont3
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont3
  %red.value.0 = load float, ptr %loadgep_2, align 4
  %red.private.value.0 = load float, ptr %2, align 4
  %7 = fadd float %red.value.0, %red.private.value.0
  store float %7, ptr %loadgep_2, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num7, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont3
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %reduce.finalize
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i32 %omp_loop.iv, %3
  %9 = mul i32 %8, 1
  %10 = add i32 %9, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  store i32 %10, ptr %loadgep_, align 4
  %11 = load i32, ptr %loadgep_, align 4
  %12 = sitofp i32 %11 to float
  %13 = load float, ptr %2, align 1
  %14 = fadd float %13, %12
  store float %14, ptr %2, align 1
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i32 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_reduction_bitwise_and() #0 {
  %structArg = alloca { ptr, ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  store i32 -1, ptr %2, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  %gep_9 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %2, ptr %gep_9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_bitwise_and..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_bitwise_and..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca i32, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  store i32 0, ptr %loadgep_, align 4
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region1
  %omp_orig = load i32, ptr %loadgep_2, align 4
  store i32 1, ptr %2, align 4
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.reduction.init
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 9, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num5, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %6
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num6)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num7, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func.3, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont3
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont3
  %red.value.0 = load i32, ptr %loadgep_2, align 4
  %red.private.value.0 = load i32, ptr %2, align 4
  %7 = and i32 %red.value.0, %red.private.value.0
  store i32 %7, ptr %loadgep_2, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num7, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont3
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %reduce.finalize
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i32 %omp_loop.iv, %3
  %9 = mul i32 %8, 1
  %10 = add i32 %9, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  store i32 %10, ptr %loadgep_, align 4
  %11 = load i32, ptr %loadgep_, align 4
  %12 = load i32, ptr %2, align 1
  %13 = and i32 %12, %11
  store i32 %13, ptr %2, align 1
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i32 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_reduction_bitwise_or() #0 {
  %structArg = alloca { ptr, ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  store i32 0, ptr %2, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  %gep_9 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %2, ptr %gep_9, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_bitwise_or..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_bitwise_or..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_2 = load ptr, ptr %gep_1, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i32, align 4
  %p.upperbound = alloca i32, align 4
  %p.stride = alloca i32, align 4
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca i32, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  store i32 0, ptr %loadgep_, align 4
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region1
  %omp_orig = load i32, ptr %loadgep_2, align 4
  store i32 0, ptr %2, align 4
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.reduction.init
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 9, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @1, i32 %omp_global_thread_num5, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  %3 = load i32, ptr %p.lowerbound, align 4
  %4 = load i32, ptr %p.upperbound, align 4
  %5 = sub i32 %4, %3
  %6 = add i32 %5, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %6
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num5)
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num6)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num7, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func.4, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont3
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont3
  %red.value.0 = load i32, ptr %loadgep_2, align 4
  %red.private.value.0 = load i32, ptr %2, align 4
  %7 = or i32 %red.value.0, %red.private.value.0
  store i32 %7, ptr %loadgep_2, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num7, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont3
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %reduce.finalize
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %8 = add i32 %omp_loop.iv, %3
  %9 = mul i32 %8, 1
  %10 = add i32 %9, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  store i32 %10, ptr %loadgep_, align 4
  %11 = load i32, ptr %loadgep_, align 4
  %12 = shl i32 1, %11
  %13 = load i32, ptr %2, align 1
  %14 = or i32 %13, %12
  store i32 %14, ptr %2, align 1
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i32 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_reduction_parallel() #0 {
  %structArg = alloca { ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  store i32 0, ptr %1, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_reduction_parallel..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_reduction_parallel..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = alloca i32, align 4
  %red.array = alloca [1 x ptr], align 8
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.par.entry
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.reduction.init

omp.reduction.init:                               ; preds = %omp.par.region
  %omp_orig = load i32, ptr %loadgep_, align 4
  store i32 0, ptr %2, align 4
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.reduction.init
  %3 = load i32, ptr %2, align 1
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %2, align 1
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp.par.region1
  %red.array.elem.0 = getelementptr inbounds [1 x ptr], ptr %red.array, i64 0, i64 0
  store ptr %2, ptr %red.array.elem.0, align 8
  %omp_global_thread_num2 = call i32 @__kmpc_global_thread_num(ptr @1)
  %reduce = call i32 @__kmpc_reduce(ptr @1, i32 %omp_global_thread_num2, i32 1, i64 8, ptr %red.array, ptr @.omp.reduction.func.5, ptr @.gomp_critical_user_.reduction.var)
  switch i32 %reduce, label %reduce.finalize [
    i32 1, label %reduce.switch.nonatomic
    i32 2, label %reduce.switch.atomic
  ]

reduce.switch.atomic:                             ; preds = %omp.region.cont
  unreachable

reduce.switch.nonatomic:                          ; preds = %omp.region.cont
  %red.value.0 = load i32, ptr %loadgep_, align 4
  %red.private.value.0 = load i32, ptr %2, align 4
  %5 = add i32 %red.value.0, %red.private.value.0
  store i32 %5, ptr %loadgep_, align 4
  call void @__kmpc_end_reduce(ptr @1, i32 %omp_global_thread_num2, ptr @.gomp_critical_user_.reduction.var)
  br label %reduce.finalize

reduce.finalize:                                  ; preds = %reduce.switch.nonatomic, %omp.region.cont
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %reduce.finalize
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #2

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) #2

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #2

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) #3

define internal void @.omp.reduction.func(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = add i32 %5, %8
  store i32 %9, ptr %4, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare i32 @__kmpc_reduce(ptr, i32, i32, i64, ptr, ptr, ptr) #3

; Function Attrs: convergent nounwind
declare void @__kmpc_end_reduce(ptr, i32, ptr) #3

define internal void @.omp.reduction.func.1(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = mul i32 %5, %8
  store i32 %9, ptr %4, align 4
  ret void
}

define internal void @.omp.reduction.func.2(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load float, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load float, ptr %7, align 4
  %9 = fadd float %5, %8
  store float %9, ptr %4, align 4
  ret void
}

define internal void @.omp.reduction.func.3(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = and i32 %5, %8
  store i32 %9, ptr %4, align 4
  ret void
}

define internal void @.omp.reduction.func.4(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = or i32 %5, %8
  store i32 %9, ptr %4, align 4
  ret void
}

define internal void @.omp.reduction.func.5(ptr %0, ptr %1) {
  %3 = getelementptr inbounds [1 x ptr], ptr %0, i64 0, i64 0
  %4 = load ptr, ptr %3, align 8
  %5 = load i32, ptr %4, align 4
  %6 = getelementptr inbounds [1 x ptr], ptr %1, i64 0, i64 0
  %7 = load ptr, ptr %6, align 8
  %8 = load i32, ptr %7, align 4
  %9 = add i32 %5, %8
  store i32 %9, ptr %4, align 4
  ret void
}

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #2

attributes #0 = { noinline }
attributes #1 = { noinline nounwind }
attributes #2 = { nounwind }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 4}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
