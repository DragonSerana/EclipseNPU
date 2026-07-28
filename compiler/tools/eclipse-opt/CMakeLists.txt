add_executable(eclipse-opt
  eclipse-opt.cpp
)

target_link_libraries(eclipse-opt PRIVATE
  MLIRExperimentalPasses 
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
  EclipseDialect 
)
