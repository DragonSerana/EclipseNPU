set(LLVM_TARGET_DEFINITIONS EclipseOps.td)
mlir_tablegen(EclipseOps.h.inc -gen-op-decls)
mlir_tablegen(EclipseOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(EclipseOpsInc)

add_mlir_dialect_library(EclipseDialect
  Dialect.cpp
  DEPENDS
  EclipseOpsInc
  LINK_COMPONENTS
  Core
  MLIRIR
)
