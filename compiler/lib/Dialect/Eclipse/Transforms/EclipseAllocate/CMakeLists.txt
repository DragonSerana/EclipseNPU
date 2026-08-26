set(ECLIPSE_ALLOC_PASSES_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include/eclipse/Dialect/Eclipse/Transforms/AllocationPasses.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_ALLOC_PASSES_TD})
mlir_tablegen(EclipseAllocationPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(EclipseAllocationPassesIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(EclipseAllocationPasses
  EclipseAllocate.cpp
  SramAllocator.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include/eclipse

  DEPENDS
  EclipseAllocationPassesIncGen
  EclipseOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  EclipseDialect
)
