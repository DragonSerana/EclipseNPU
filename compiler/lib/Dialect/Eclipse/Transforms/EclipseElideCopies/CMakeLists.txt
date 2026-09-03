set(ECLIPSE_ELIDE_COPIES_PASSES_TD
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include/eclipse/Dialect/Eclipse/Transforms/ElideCopiesPasses.td)
set(LLVM_TARGET_DEFINITIONS ${ECLIPSE_ELIDE_COPIES_PASSES_TD})
mlir_tablegen(EclipseElideCopiesPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(EclipseElideCopiesPassesIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(EclipseElideCopies
  EclipseElideCopies.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include/eclipse

  DEPENDS
  EclipseElideCopiesPassesIncGen
  EclipseOpsInc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  EclipseDialect
)
