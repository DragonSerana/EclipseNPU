# H2.2 / H2.3 acceptance

Acceptance record for the matmul main line. Numbers are from the current pipeline
(`--one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse
--eclipse-elide-copies --canonicalize --eclipse-allocate --eclipse-to-easm`).

## H2.2 - fixed tile-K matmul, 128x128x128

- tile-K = 16, 8 K-blocks. Block 0 is peeled (`accumulate=false`), the rest run in an `scf.for`
  (`accumulate=true`).
- The emitter statically unrolls the loop.

| config | SRAM layout | cycle | hazard | vs PyTorch |
|--------|-------------|------:|--------|------------|
| `golden-mirror` | A=`0x10000000` B=`0x10008000` C=`0x10010000` | 11536 | 0 | 6.109e-04 PASS |
| `bump` | sequential | 11536 | 0 | 6.109e-04 PASS |

The `golden-mirror` `.easm` differs from `tests/golden/golden.easm` by exactly one line: the golden adds
a redundant trailing `SYNC` before the final `DMA_STORE` that the compiler does not emit. `SYNC` costs
0 cycles in the cmodel, so the total is still 11536.

## H2.3 - elementwise fusion (store->load elision)

`eclipse-elide-copies` removes a `dma_store` -> `dma_load` round-trip when the intermediate DDR buffer
has no other consumer. This keeps the elementwise chain resident in SRAM instead of spilling C to DDR
and reloading it.

Worked at both 16x16 and 128x128. Full e2e (generate inputs -> compile -> Simulator -> compare):

| case | cosine | max rel err | cycle | hazard |
|------|-------:|------------:|------:|--------|
| matmul_128 | 0.999999 | 6.109e-04 | 11536 | 0 |
| matmul_add_128 | 0.999998 | 6.002e-04 | 12704 | 0 |
| matmul_add_relu_128 | 0.999999 | 7.262e-04 | 12832 | 0 |
| matmul_16 | 1.000000 | 0.000e+00 | 112 | 0 |
| matmul_add_16 | 1.000000 | 0.000e+00 | 146 | 0 |
| matmul_add_relu_16 | 1.000000 | 0.000e+00 | 148 | 0 |

Criteria: `cosine >= 0.999` and `max rel err < 1e-2`. All cases pass; hazard check is 0 violations on
every generated `.easm`. Inputs are generated with a fixed seed (`tools/gen_inputs.py --seed 0`), so any
failure reproduces.

The fused chain (matmul -> bias add -> relu) stays in SRAM end to end: matmul result is consumed
directly by the add, and the add result directly by the relu, with a single `DMA_STORE` at the end.

## how to re-run

```
source env.sh
Eclipse-test            # check-eclipse + check-golden + check-accuracy
# or
python3 tools/accuracy_check.py --seed 0
```
