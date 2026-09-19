add_library(EclipseRuntime
  lib/cmodel.cpp
  lib/cmodel_exec.cpp
  lib/cmodel_cycles.cpp
  lib/simulator.cpp
)

target_include_directories(EclipseRuntime PUBLIC include)
