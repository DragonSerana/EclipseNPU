# Simulator roofline (H2)

Roofline for the matmul main line on the cmodel. Only the simulator is analyzed here; the
real-RDNA4 counterpart is a separate GPU work item.

## method

Parse the generated `.easm` and recompute cycles with the same formulas the cmodel uses
(`runtime/lib/cmodel.cpp` `computeCycles`):

```
MATMUL            ceilDiv(M*N / MAC_PER_CYCLE) * K
DMA_LOAD/STORE    ceilDiv(bursts * DMA_BURST_BYTES / DMA_BYTES_PER_CYCLE) + DMA_FIXED_OVERHEAD
ELEMENTWISE_ADD/ACT  ceilDiv(n / ELEM_PER_CYCLE)
SYNC              0
```

Constants (`runtime/include/eclipse_isa.h`):

| | |
|---|---|
| `MAC_PER_CYCLE` | 256 |
| `DMA_BYTES_PER_CYCLE` | 32 |
| `DMA_BURST_BYTES` | 16 |
| `DMA_FIXED_OVERHEAD` | 16 |
| `ELEM_PER_CYCLE` | 128 |
| `DTYPE_SIZE` | 2 |

## results (128x128x128)

| chain | MATMUL cyc | DMA cyc | EW/ACT cyc | total | FLOP/byte | MAC util | DMA util | bounded by |
|-------|-----------:|--------:|-----------:|------:|----------:|---------:|---------:|------------|
| matmul | 8192 | 3344 | 0 | 11536 | 42.7 | 71% | 29% | compute |
| matmul_add | 8192 | 4384 | 128 | 12704 | 32.0 | 64% | 35% | compute |
| matmul_add_relu | 8192 | 4384 | 256 | 12832 | 32.0 | 64% | 34% | compute |

Ridge point (compute peak / memory peak) = `MAC_PER_CYCLE*2 / DMA_BYTES_PER_CYCLE` = `512/32`
= **16.0 FLOP/byte**.

## conclusion

All three chains are **compute-bound (MAC-limited)**. Arithmetic intensity (32-42.7 FLOP/byte)
is well above the 16 FLOP/byte ridge, so the MAC engine is the limiter, not the DMA copy engine.

The tile-K=16 blocking reloads A and B eight times, but the DMA engine is fast enough (32 B/cyc)
that those copies never become the bottleneck. On this simulator the matmul is limited by the
256 MAC/cycle engine, not by bandwidth.

For the simulator this means double-buffering / bigger tiles won't move the needle on the matmul
cycles themselves (they'd only help what is already not the bottleneck). The MAC engine is the
ceiling. Reaching ~100% MAC util on the multiply isn't happening here because the load / store /
sync overhead and the elementwise tail take the remaining 29-36%.
