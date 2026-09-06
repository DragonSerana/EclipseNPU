# tools/：可复用的测试工具与 .easm -> simulator 驱动器。
add_executable(eclipse-run eclipse-run.cpp)
target_link_libraries(eclipse-run PRIVATE EclipseRuntime)
