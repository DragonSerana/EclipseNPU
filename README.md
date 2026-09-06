# EclipseNPU

A small NPU toolchain. An MLIR-based compiler lowers `linalg` (matmul and elementwise ops) to a
custom textual ISA (`.easm`), which is then executed on a cycle-accurate simulator. There is no
generic CPU/GPU codegen: the instruction queue is the backend.

The ISA is documented in `docs/spec/isa-v0.1.md`. The compiler targets a hand-written golden kernel
(`tests/golden/matmul_golden.cpp`), and a static hazard checker validates generated programs.

## build

```bash
source env.sh
Eclipse-build
```

Requires a self-built LLVM/MLIR (see `env.sh` for the pinned paths and commit). The main binaries land
in `build/bin/`:

- `eclipse-opt` - the compiler driver
- `eclipse-run` - read a `.easm`, drive the simulator, dump the output

## quick start

```bash
source env.sh

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
source env.sh
Eclipse-test
```

This runs three things:

- `check-eclipse` - lit tests for the dialect and passes
- `check-golden` - matches the compiler against the golden reference
- `check-accuracy` - end-to-end numeric comparison against PyTorch

Accuracy uses cosine similarity (`>= 0.999`) and a normalized max relative error (`< 1e-2`). Inputs are
generated with a fixed seed, so failures are reproducible.

## layout

```
compiler/    MLIR dialect + passes (convert, allocate, elide-copies, to-easm)
runtime/     cmodel + simulator (the .easm interpreter)
tests/       lit tests, golden reference, e2e case inputs
tools/       test scripts + the .easm runner
docs/        specs, notes, plans, reports
```

## status

H2 of the roadmap is the matmul main line, and the current pipeline covers:

- tile-K matmul to 128x128x128 (8 K-blocks, `accumulate` peeling)
- elementwise fusion: matmul -> bias add -> relu stays resident in SRAM (no DDR round-trip)
- both a bump SRAM allocator and a `golden-mirror` layout for regression diffs

See `docs/report/acceptance-h2.md` for the numbers. H3 (argmax / attention) and the real-GPU
(RDNA4) side are on the roadmap (`docs/plans/roadmap.md`).
