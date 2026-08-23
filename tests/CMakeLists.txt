add_executable(matmul_golden golden/matmul_golden.cpp)
target_link_libraries(matmul_golden PRIVATE EclipseRuntime)

find_program(PYTHON3_EXECUTABLE python3)

find_program(LLVM_LIT_EXECUTABLE NAMES lit lit.py)
if(NOT LLVM_LIT_EXECUTABLE)
  message(FATAL_ERROR "lit not found; install it with: pip3 install --user lit")
endif()

add_custom_target(check-eclipse
  COMMAND ${LLVM_LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_SOURCE_DIR}/lit
  DEPENDS eclipse-opt matmul_golden
  USES_TERMINAL
)

add_custom_target(check-golden
  COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/golden/matmul_check.py
  COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/hazard_check.py
          ${CMAKE_CURRENT_SOURCE_DIR}/golden/golden.easm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/golden
  DEPENDS matmul_golden
  USES_TERMINAL
)
