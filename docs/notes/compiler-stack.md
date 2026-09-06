# Compiler stack

How the MLIR pipeline turns `linalg` into `.easm`. The stack is deliberately small:
lowering is a straight-line sequence of passes, no `transform` dialect, no `affine`, no `vector`.

## Pipeline

```
eclipse-opt input.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-eclipse \
  --eclipse-elide-copies \
  --canonicalize \
  --eclipse-allocate="layout=bump|golden-mirror" \
  --eclipse-to-easm="output-easm=out.easm"
```

Each pass:

| pass | job |
|------|-----|
| `one-shot-bufferize` | tensor -> memref, with function boundaries. The only real dependency on upstream bufferization. |
| `convert-linalg-to-eclipse` | linalg.matmul / elementwise add / single-in-out generic (treated as relu) -> eclipse ops. Splits K on matmul. Rewrites function args to memory space 1 (DDR ABI). |
| `eclipse-elide-copies` | store->load forwarding. Removes a `dma_store %s -> %ddr` + `dma_load %ddr -> %t` round-trip when `%ddr` has no other consumer, rewiring `%t`'s readers to `%s`. |
| `canonicalize` | cleans the dead allocs / casts / subviews that the elision leaves behind. |
| `eclipse-allocate` | assigns SRAM addresses and DDR ABI addresses. `layout=bump` is a sequential allocator; `golden-mirror` pins the three matmul buffers to the golden layout. |
| `eclipse-to-easm` | textual serializer. Emits one instruction per line, program order. |

## key decisions

**tile-K lives in the conversion pattern.** `MatmulLowering` peels block 0 (`accumulate=false`) then
emits an `scf.for 1..nBlocks` (`accumulate=true`). No `linalg::tileLinalgOp`, no transform dialect. The
`tileK` constant is 16.

**The emitter unrolls `scf.for` itself.** `--eclipse-to-easm` runs `loopUnrollFull` then walks the ops,
substituting the induction variable and resolving `memref.subview` offsets/strides to concrete addresses.

**Residency is decided at the eclipse layer, not in linalg.** A buffer stays in SRAM or is spilled to DDR
based on the ops that touch it. That's why `eclipse-elide-copies` exists as a normalization pass after
lowering instead of being folded into a linalg fusion. A linalg-level `tile-and-fuse` would be useless
here: each `linalg` op still lowers to its own `dma_load`/`dma_store`.

**golden-mirror is a regression anchor, not the endgame.** `layout=golden-mirror` reproduces the
hand-written golden's three fixed SRAM buffers (A=`0x10000000`, B=`0x10008000`, C=`0x10010000`). It is
used to diff the compiler output against `tests/golden/golden.easm`. The real allocator is `bump`; a
liveness-based allocator is the eventual goal.

## memory model

- memory space `0` = SRAM (the SIMD compute side), `1` = DDR (the ABI side). `EclipseConstants.h`.
- `SRAM_ALIGNMENT = 32`.
- DDR ABI: `arg0 @ 0x80010000`, `arg1 @ 0x80020000`, ... assigned by `eclipse-allocate`.
- `memref.dealloc` is not emitted to `.easm`; it only marks lifetime in the IR.

## notes / gotchas

- `memref.subview` strides in the IR are index advances (1 for a packed row-major slice). The memory
  stride the DMA payload wants is `base.cols * DTYPE_SIZE` (packed assumption).
- This MLIR build has `arith.maximumf`, not `arith.maxf`. A relu generic body uses `maximumf`.
- Subview offsets are folded to constants by the time the emitter runs (via `loopUnrollFull` +
  `tryToFold`). Dynamic offsets (loop induction var) don't reach the emitter.
