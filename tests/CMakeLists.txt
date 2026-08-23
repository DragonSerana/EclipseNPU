add_executable(matmul_golden matmul_golden.cpp)
target_link_libraries(matmul_golden PRIVATE EclipseRuntime)

find_program(LLVM_LIT_EXECUTABLE NAMES lit lit.py)
if(NOT LLVM_LIT_EXECUTABLE)
  message(FATAL_ERROR "lit not found; install it with: pip3 install --user lit")
endif()

add_custom_target(check-eclipse
  COMMAND ${LLVM_LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_SOURCE_DIR}/lit
  DEPENDS eclipse-opt matmul_golden
  USES_TERMINAL
)
