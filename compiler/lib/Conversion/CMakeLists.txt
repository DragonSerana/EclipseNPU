set(ECLIPSE_PASSES_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/eclipse/Conversion/Passes.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_PASSES_TD})
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(EclipseConversionPassesIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(EclipseConversionPasses
  ConvertLinalgToEclipse.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/eclipse

  DEPENDS
  EclipseConversionPassesIncGen
  EclipseOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRLinalgDialect
  EclipseDialect
)
