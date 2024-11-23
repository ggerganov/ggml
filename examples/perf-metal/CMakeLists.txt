#
# perf-metal

set(TEST_TARGET perf-metal)
add_executable(${TEST_TARGET} perf-metal.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

