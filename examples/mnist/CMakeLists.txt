#
# mnist

set(TEST_TARGET mnist)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# mnist-cpu

set(TEST_TARGET mnist-cpu)
add_executable(${TEST_TARGET} main-cpu.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
