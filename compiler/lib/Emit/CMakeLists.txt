set(ECLIPSE_EMIT_PASSES_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/eclipse/Emit/Passes.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_EMIT_PASSES_TD})
mlir_tablegen(EclipseEmitPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(EclipseEmitPassesIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(EclipseEmitPasses
  EclipseToEasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/eclipse

  DEPENDS
  EclipseEmitPassesIncGen
  EclipseOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  EclipseDialect
)
