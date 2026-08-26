include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Eclipse)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Eclipse/Transforms/EclipseAllocate)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Eclipse/Transforms/EclipseToEasm)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Conversion)

add_subdirectory(lib/Dialect/Eclipse)
add_subdirectory(lib/Conversion)
add_subdirectory(tools/eclipse-opt)
