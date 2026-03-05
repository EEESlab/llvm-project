; ModuleID = '/home/lucap/eeeslab/llvm-project/clang/test/CIR/CodeGenOpenMP/master/pragma-omp-master.c'
source_filename = "/home/lucap/eeeslab/llvm-project/clang/test/CIR/CodeGenOpenMP/master/pragma-omp-master.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

declare void @use(i32)

; Function Attrs: noinline
define dso_local void @test_master() #0 {
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 0, ptr @test_master..omp_par)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_master..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr) #1 {
omp.par.entry:
  %tid.addr.local = alloca i32, align 4
  %0 = load i32, ptr %tid.addr, align 4
  store i32 %0, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.par.entry
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  %omp_global_thread_num2 = call i32 @__kmpc_global_thread_num(ptr @1)
  %1 = call i32 @__kmpc_master(ptr @1, i32 %omp_global_thread_num2)
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %omp_region.body, label %omp_region.end

omp_region.end:                                   ; preds = %omp.par.region1, %omp_region.finalize
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp_region.end
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_region.body:                                  ; preds = %omp.par.region1
  br label %omp.master.region

omp.master.region:                                ; preds = %omp_region.body
  call void @use(i32 42)
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp.master.region
  br label %omp_region.finalize

omp_region.finalize:                              ; preds = %omp.region.cont3
  call void @__kmpc_end_master(ptr @1, i32 %omp_global_thread_num2)
  br label %omp_region.end

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: noinline
define dso_local void @test_master_standalone() #0 {
  %structArg = alloca { ptr }, align 8
  %1 = alloca i32, i64 1, align 4
  br label %entry

entry:                                            ; preds = %0
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_ = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
  store ptr %1, ptr %gep_, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @test_master_standalone..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: noinline nounwind
define internal void @test_master_standalone..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #1 {
omp.par.entry:
  %gep_ = getelementptr { ptr }, ptr %0, i32 0, i32 0
  %loadgep_ = load ptr, ptr %gep_, align 8, !align !1
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.par.entry
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  %omp_global_thread_num2 = call i32 @__kmpc_global_thread_num(ptr @1)
  %2 = call i32 @__kmpc_master(ptr @1, i32 %omp_global_thread_num2)
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %omp_region.body, label %omp_region.end

omp_region.end:                                   ; preds = %omp.par.region1, %omp_region.finalize
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp_region.end
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_region.body:                                  ; preds = %omp.par.region1
  br label %omp.master.region

omp.master.region:                                ; preds = %omp_region.body
  store i32 1, ptr %loadgep_, align 4
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp.master.region
  br label %omp_region.finalize

omp_region.finalize:                              ; preds = %omp.region.cont3
  call void @__kmpc_end_master(ptr @1, i32 %omp_global_thread_num2)
  br label %omp_region.end

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #2

; Function Attrs: nounwind
declare i32 @__kmpc_master(ptr, i32) #2

; Function Attrs: nounwind
declare void @__kmpc_end_master(ptr, i32) #2

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #2

attributes #0 = { noinline }
attributes #1 = { noinline nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 4}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
