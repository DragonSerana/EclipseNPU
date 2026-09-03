add_executable(eclipse-opt
  eclipse-opt.cpp
)

target_link_libraries(eclipse-opt PRIVATE
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRAsmParser
  MLIROptLib
  MLIRPass
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  EclipseDialect
  EclipseConversionPasses
  EclipseAllocationPasses
  EclipseEmitPasses
  EclipseElideCopies
)
