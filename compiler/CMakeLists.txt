include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Eclipse)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Conversion)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Allocation)

add_subdirectory(lib/Dialect/Eclipse)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/Allocation)
add_subdirectory(tools/eclipse-opt)
