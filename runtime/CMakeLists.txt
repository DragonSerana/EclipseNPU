add_library(EclipseRuntime
  lib/cmodel.cpp
  lib/simulator.cpp
)

target_include_directories(EclipseRuntime PUBLIC include)
