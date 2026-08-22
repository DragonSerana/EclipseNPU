set(ECLIPSE_OPS_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/eclipse/Dialect/Eclipse/EclipseOps.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_OPS_TD})
mlir_tablegen(EclipseOps.h.generated.inc -gen-op-decls)
mlir_tablegen(EclipseOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(EclipseOpsInc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_dialect_library(EclipseDialect
  EclipseDialect.cpp
  Ops/DmaLoad.cpp
  DEPENDS
  EclipseOpsInc
)
