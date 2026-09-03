#ifndef ECLIPSE_ELIDE_COPIES_PASSES
#define ECLIPSE_ELIDE_COPIES_PASSES

include "mlir/Pass/PassBase.td"

def EclipseElideCopies : Pass<"eclipse-elide-copies", "::mlir::ModuleOp"> {
  let summary = "Elide DMA store->load copies that round-trip through DDR";
  let description = [{
    After conversion, fold a `eclipse.dma_store %s -> %ddr` sequence back into a
    `eclipse.dma_load %ddr -> %t` when `%ddr` has no other users, so elementwise
    chains stay resident in SRAM instead of spilling to DDR and back. The user of
    `%t` is rewired to `%s` and the store/load pair are removed.

    Mirrors IREE's `elide-async-copies`; residency-vs-spill is decided at the
    eclipse layer, not at linalg level (so this must run AFTER lowering, never
    as a linalg tile-and-fuse).
  }];

  let dependentDialects = [
    "mlir::eclipse::EclipseDialect"
  ];
}

#endif // ECLIPSE_ELIDE_COPIES_PASSES
