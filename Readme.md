# EclipseNPU

EclipseNPU is a full-stack NPU design project: instruction set, MLIR-based compiler, operator kernels, and a cycle-level simulator. The goal is to run a small LLM (Qwen2.5-0.5B, ~0.5B parameters) end-to-end on the designed NPU, with every numerical result checked against a PyTorch reference. The same compilation/tiling methodology is also validated on real hardware (AMD RDNA4) as an auxiliary track.

## Status

Phase 1 (ISA v0.1 + cmodel/simulator) is complete; Phase 2 (toolchain) is in progress.

- ISA v0.1 frozen — see [docs/spec/isa-v0.1.md](docs/spec/isa-v0.1.md)
- cmodel: all six instructions implemented (DMA_LOAD/STORE, MATMUL, ELEMENTWISE_ADD, ACT, SYNC); MATMUL uses fp16 in → fp32 block accumulation → fp16 write-back
- simulator: sequential instruction queue with cycle accounting (`computeCycles`: MAC throughput + DMA bandwidth model)
- verification: hand-written 128×128 matmul instruction stream (K-tiled, accumulate) checked against PyTorch fp16 / fp64 — see [docs/spec/accuracy.md](docs/spec/accuracy.md) and [docs/report/cycle-report-h1.md](docs/report/cycle-report-h1.md)
- compiler: skeleton only (Eclipse dialect, `eclipse-opt` stub); H2 = 6 leaf ops + linalg lowering chain + lit wired into the build
- tests: `matmul_golden` built by the main build; lit tests present, not yet run under CI

The milestone plan is in [docs/plans/roadmap.md](docs/plans/roadmap.md).

## ISA v0.1

- Data type: fp16 (16-bit, signed)
- Instruction format: fixed 8 bytes (`opcode: u32` + `desc_ptr: u32`); descriptors live in the command queue region
- Memory model:
  - SRAM: 512 KB at `0x10000000`
  - DDR: 1 GB at `0x80000000`; top 64 KB reserved as the command queue
  - tensor buffers are 16-byte aligned (compiler guarantees)
- Instructions:
  - `DMA_LOAD` / `DMA_STORE`: strided 2-D tile transfer
  - `MATMUL`: `M×K · K×N`, fp32 block accumulation, fp16 write-back; `accumulate` flag selects overwrite vs. accumulate
  - `ELEMENTWISE_ADD`
  - `ACT` (ReLU)
  - `SYNC` (fence)
- Compute operands must be packed in SRAM; only DMA supports strides

## Repository Layout

| Directory  | Contents |
| ---------- | -------- |
| `docs/`    | plans, spec, notes, reports |
| `runtime/` | `cmodel` (SRAM/DDR model + instruction interpretation), `simulator` (cycle counting) |
| `compiler/`| Eclipse dialect (MLIR), `eclipse-opt` |
| `tests/`   | golden 对拍 + lit 测试 |

## Build

Requirements: CMake >= 3.15.4, LLVM/MLIR (paths exported as `LLVM_DIR` / `MLIR_DIR`).

```
source env.sh
Eclipse-build
```

`env.sh` sets up the LLVM/MLIR paths, adds `build/bin` to `PATH`, and provides `Eclipse-build` / `Eclipse-clean` / `Eclipse-format` helpers. Outputs land in `build/bin` (e.g. `eclipse-opt`).

## Roadmap

- H1: ISA v0.1 complete — hand-written matmul instruction stream checked against PyTorch, cycle model implemented
- H2: linalg → Eclipse lowering pipeline + lit wired into CI; auxiliary track: hand-written HIP GEMM on RDNA4 with roofline report
- H3: operator audit ([docs/ops-audit.md](docs/ops-audit.md)) → ISA v0.2 freeze (EWISE_MUL, DIV/RSQRT, fp32 ACC, async DMA, ...) → matmul / argmax / attention, hand-written golden vs. compiler-generated
- H4: Qwen2.5-0.5B end-to-end on the simulator (prefill 128 + decode 16), logits checked against a PyTorch fp16 reference
- H5: GPU deep dive — MLIR CodeGen backends for RDNA4, dual-backend roofline comparison

## License

Not yet selected.
