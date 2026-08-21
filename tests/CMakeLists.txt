add_executable(matmul_golden matmul_golden.cpp)
target_link_libraries(matmul_golden PRIVATE EclipseRuntime)

find_program(PYTHON3_EXECUTABLE python3)

add_custom_target(check-eclipse
  COMMAND ${PYTHON3_EXECUTABLE} -m lit -sv ${CMAKE_CURRENT_SOURCE_DIR}/lit
  DEPENDS eclipse-opt matmul_golden
  USES_TERMINAL
)
