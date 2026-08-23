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
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  EclipseDialect
  EclipseConversionPasses
)
