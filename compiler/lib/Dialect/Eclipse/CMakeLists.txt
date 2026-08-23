set(ECLIPSE_OPS_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/eclipse/Dialect/Eclipse/EclipseOps.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_OPS_TD})
mlir_tablegen(EclipseOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(EclipseOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(EclipseOps.h.generated.inc -gen-op-decls)
mlir_tablegen(EclipseOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(EclipseOpsInc)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_dialect_library(EclipseDialect
  EclipseDialect.cpp
  Ops/DmaLoad.cpp
  Ops/DmaStore.cpp
  Ops/Matmul.cpp
  Ops/EwiseAdd.cpp
  Ops/Act.cpp
  Ops/Sync.cpp
  DEPENDS
  EclipseOpsInc
)
