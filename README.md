# EclipseNPU

EclipseNPU is a full-stack NPU design project: instruction set, MLIR-based compiler, operator
kernels, and a cycle-level simulator. The goal is to run a small LLM (Qwen2.5-0.5B, ~0.5B
parameters) end-to-end on the designed NPU, with every numerical result checked against a PyTorch
reference. The same compilation/tiling methodology is also validated on real hardware (AMD RDNA4)
as an auxiliary track.

There is no generic CPU/GPU codegen: the instruction queue is the backend.

## status

- **H1 (done)** — ISA v0.1 + cmodel/simulator. Six instructions implemented, `computeCycles`
  (MAC throughput + DMA bandwidth model). A hand-written 128×128 K-tiled matmul instruction stream is
  checked against PyTorch. See [docs/spec/isa.md](docs/spec/isa.md).
- **H2 (current)** — linalg → Eclipse lowering chain, `eclipse-opt` → `.easm` → `eclipse-run`, lit
  wired into the build, e2e accuracy + hazard checks. Covers tile-K matmul to 128×128×128 and
  elementwise fusion (matmul → bias add → relu, kept resident in SRAM).
- The rest is on the [roadmap](docs/plans/roadmap.md).

## build

```bash
source scripts/env.sh
Eclipse-build
```

Requires a self-built LLVM/MLIR (see `scripts/env.sh` for the pinned paths and commit). The main
binaries land in `build/bin/`:

- `eclipse-opt` - the compiler driver
- `eclipse-run` - read a `.easm`, drive the simulator, dump the output

## quick start

```bash
source scripts/env.sh

# compile a model to .easm
Eclipse-compile tests/e2e/matmul_add_128.mlir m.easm

# generate inputs, run it, compare against PyTorch
python3 tools/gen_inputs.py build/tmp --M 128 --N 128 --K 128 --bias --seed 0
Eclipse-inference m.easm m.raw build/tmp/a.raw build/tmp/b.raw build/tmp/bias.raw
Eclipse-check m.raw build/tmp/a.raw build/tmp/b.raw --M 128 --N 128 --K 128 --bias build/tmp/bias.raw
```

The compile pipeline is

```
one-shot-bufferize -> convert-linalg-to-eclipse -> eclipse-elide-copies
-> canonicalize -> eclipse-allocate -> eclipse-to-easm
```

## test

```bash
source scripts/env.sh
Eclipse-test
```

This runs three things:

- `check-eclipse` - lit tests for the dialect and passes
- `check-golden` - matches the compiler against the golden reference
- `check-accuracy` - end-to-end numeric comparison against PyTorch

Accuracy uses cosine similarity (`>= 0.999`) and a normalized max relative error (`< 1e-2`).
Inputs are generated with a fixed seed, so failures are reproducible.

## ISA v0.1

- fp16 data type, fixed 8-byte instruction (`opcode: u32` + `desc_ptr: u32`); descriptors live in the
  command queue region.
- Memory model: SRAM 512 KB at `0x10000000`; DDR 1 GB at `0x80000000` (top 64 KB reserved as the
  command queue); tensor buffers are 16-byte aligned.
- Instructions: `DMA_LOAD`/`DMA_STORE` (strided 2-D tile), `MATMUL` (`M×K · K×N`, fp32 block
  accumulation, fp16 write-back, `accumulate` flag), `ELEMENTWISE_ADD`, `ACT` (ReLU), `SYNC`.
- Compute operands must be packed in SRAM; only DMA supports strides. Full spec:
  [docs/spec/isa.md](docs/spec/isa.md).

## layout

```
compiler/    MLIR dialect + passes (convert, allocate, elide-copies, to-easm)
runtime/     cmodel + simulator (the .easm interpreter)
tests/       lit tests, golden reference, e2e case inputs
tools/       test scripts + the .easm runner
scripts/     env + CI scripts
docs/        specs, notes, plans, reports
```

## roadmap

- H3: operator audit → ISA v0.2 (EWISE_MUL, DIV/RSQRT, fp32 ACC, async DMA, ...) → matmul / argmax /
  attention, hand-written golden vs. compiler-generated.
- H3.5: single decoder layer end-to-end
- H4: Qwen2.5-0.5B on the simulator (prefill 128 + decode 16), logits checked against a PyTorch fp16
  reference
- H5: MLIR CodeGen for RDNA4, dual-backend roofline comparison

## license

Not yet selected.
