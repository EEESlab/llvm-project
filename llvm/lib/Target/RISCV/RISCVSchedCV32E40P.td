//==- RISCVSchedCV32E40P.td - OpenHW CV32E40P Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CV32E40P: https://github.com/openhwgroup/cv32e40p
// Timings from the user manual, "Pipeline Details":
// https://docs.openhwfoundation.org/projects/cv32e40p-user-manual/en/latest/pipeline.html
//
// 4-stage, single-issue, in-order pipeline. Results are forwarded from EX to
// ID, so a dependent ALU/MUL instruction issues the next cycle. Load data is
// only available one cycle later: an instruction that uses a load result
// immediately after the load stalls for one cycle, hence load latency 2.
// The post-increment address of the CORE-V loads/stores is computed by the
// ALU in EX and forwarded like any ALU result (latency 1).

defvar CV32E40PFPUAddMulLat = 1;
defvar CV32E40PFPUOthersLat = 0;

def CV32E40PModel : SchedMachineModel {
  let MicroOpBufferSize = 0; // In-order.
  let IssueWidth = 1;        // Single issue.
  let LoadLatency = 2;
  let MispredictPenalty = 2; // A taken branch costs 3 cycles instead of 1.
  let PostRAScheduler = true;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVInstructions];
}

let SchedModel = CV32E40PModel in {

let BufferSize = 0 in {
def CV32E40P_ALU  : ProcResource<1>;
def CV32E40P_LSU  : ProcResource<1>;
def CV32E40P_MUL  : ProcResource<1>;
def CV32E40P_DIV  : ProcResource<1>;
def CV32E40P_CFU  : ProcResource<1>;
def CV32E40P_FPU  : ProcResource<1>;
def CV32E40P_FDIV : ProcResource<1>;
}

// Branching
def : WriteRes<WriteJmp, [CV32E40P_CFU]>;
def : WriteRes<WriteJal, [CV32E40P_CFU]>;
def : WriteRes<WriteJalr, [CV32E40P_CFU]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [CV32E40P_ALU]>;
def : WriteRes<WriteIALU, [CV32E40P_ALU]>;
def : WriteRes<WriteShiftImm32, [CV32E40P_ALU]>;
def : WriteRes<WriteShiftImm, [CV32E40P_ALU]>;
def : WriteRes<WriteShiftReg32, [CV32E40P_ALU]>;
def : WriteRes<WriteShiftReg, [CV32E40P_ALU]>;

// Integer multiplication: mul is single-cycle, the upper-word variants take
// 5 cycles and block the multiplier.
def : WriteRes<WriteIMul, [CV32E40P_MUL]>;
def : WriteRes<WriteIMul32, [CV32E40P_MUL]>;
def CV32E40PWriteIMulH : SchedWriteRes<[CV32E40P_MUL]> {
  let Latency = 5;
  let ReleaseAtCycles = [5];
}
def : InstRW<[CV32E40PWriteIMulH, ReadIMul, ReadIMul],
             (instrs MULH, MULHU, MULHSU)>;

// Integer division/remainder: 3 to 35 cycles depending on the operands;
// modelled with the worst case.
let Latency = 35, ReleaseAtCycles = [35] in {
def : WriteRes<WriteIDiv32, [CV32E40P_DIV]>;
def : WriteRes<WriteIDiv, [CV32E40P_DIV]>;
def : WriteRes<WriteIRem32, [CV32E40P_DIV]>;
def : WriteRes<WriteIRem, [CV32E40P_DIV]>;
}

// Memory. One access per cycle (a misaligned access crossing a word boundary
// takes two). Loads have a one-cycle load-use penalty.
let Latency = 2 in {
def : WriteRes<WriteLDB, [CV32E40P_LSU]>;
def : WriteRes<WriteLDH, [CV32E40P_LSU]>;
def : WriteRes<WriteLDW, [CV32E40P_LSU]>;
def : WriteRes<WriteLDD, [CV32E40P_LSU]>;
def : WriteRes<WriteFLD32, [CV32E40P_LSU]>;
}
def : WriteRes<WriteSTB, [CV32E40P_LSU]>;
def : WriteRes<WriteSTH, [CV32E40P_LSU]>;
def : WriteRes<WriteSTW, [CV32E40P_LSU]>;
def : WriteRes<WriteSTD, [CV32E40P_LSU]>;
def : WriteRes<WriteFST32, [CV32E40P_LSU]>;

// CORE-V XCVmem. These instructions carry no scheduling information of their
// own. A post-increment access also writes back its base register, which is
// available the next cycle. That write-back is part of the same single-issue
// instruction, so it must not count as a micro-op: with IssueWidth = 1, an
// extra micro-op would make every post-increment access take two issue
// cycles in the model.
def CV32E40PWriteLoad : SchedWriteRes<[CV32E40P_LSU]> { let Latency = 2; }
def CV32E40PWriteStore : SchedWriteRes<[CV32E40P_LSU]>;
def CV32E40PWriteAddrInc : SchedWriteRes<[]> { let NumMicroOps = 0; }

// rd, rs1_wb <- (rs1), imm12
def : InstRW<[CV32E40PWriteLoad, CV32E40PWriteAddrInc, ReadMemBase],
             (instregex "^CV_L(B|BU|H|HU|W)_ri_inc$")>;
// rd, rs1_wb <- (rs1), rs2
def : InstRW<[CV32E40PWriteLoad, CV32E40PWriteAddrInc, ReadMemBase, ReadIALU],
             (instregex "^CV_L(B|BU|H|HU|W)_rr_inc$")>;
// rd <- rs2(rs1)
def : InstRW<[CV32E40PWriteLoad, ReadMemBase, ReadIALU],
             (instregex "^CV_L(B|BU|H|HU|W)_rr$")>;
// rs1_wb <- rs2, (rs1), imm12
def : InstRW<[CV32E40PWriteAddrInc, CV32E40PWriteStore, ReadStoreData,
              ReadMemBase],
             (instregex "^CV_S(B|H|W)_ri_inc$")>;
// rs1_wb <- rs2, (rs1), rs3
def : InstRW<[CV32E40PWriteAddrInc, CV32E40PWriteStore, ReadStoreData,
              ReadMemBase, ReadIALU],
             (instregex "^CV_S(B|H|W)_rr_inc$")>;
// rs2 -> rs3(rs1)
def : InstRW<[CV32E40PWriteStore, ReadStoreData, ReadMemBase, ReadIALU],
             (instregex "^CV_S(B|H|W)_rr$")>;

// Single-precision floating point (F or Zfinx). The FPU is pipelined for
// add/multiply/fused multiply-add; divide and square root are iterative.
let Latency = !add(CV32E40PFPUAddMulLat, 1) in {
def : WriteRes<WriteFAdd32, [CV32E40P_FPU]>;
def : WriteRes<WriteFMul32, [CV32E40P_FPU]>;
def : WriteRes<WriteFMA32, [CV32E40P_FPU]>;
}
let Latency = !add(CV32E40PFPUOthersLat, 1) in {
def : WriteRes<WriteFSGNJ32, [CV32E40P_FPU]>;
def : WriteRes<WriteFMinMax32, [CV32E40P_FPU]>;
def : WriteRes<WriteFCvtI32ToF32, [CV32E40P_FPU]>;
def : WriteRes<WriteFCvtF32ToI32, [CV32E40P_FPU]>;
def : WriteRes<WriteFClass32, [CV32E40P_FPU]>;
def : WriteRes<WriteFCmp32, [CV32E40P_FPU]>;
def : WriteRes<WriteFMovF32ToI32, [CV32E40P_FPU]>;
def : WriteRes<WriteFMovI32ToF32, [CV32E40P_FPU]>;
}
// Divide/square root: up to 19 cycles, modelled with the worst case.
let Latency = 19, ReleaseAtCycles = [19] in {
def : WriteRes<WriteFDiv32, [CV32E40P_FDIV]>;
def : WriteRes<WriteFSqrt32, [CV32E40P_FDIV]>;
}
// RV64-only conversions, never selected on this RV32 core.
let Unsupported = true in {
def : WriteRes<WriteFCvtI64ToF32, []>;
def : WriteRes<WriteFCvtF32ToI64, []>;
}

// Others
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypasses (none)
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedA;
defm : UnsupportedSchedD;
defm : UnsupportedSchedZfhmin;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsf;
}
