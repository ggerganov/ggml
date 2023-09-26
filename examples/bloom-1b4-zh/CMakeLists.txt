#
# bloom-1b4-zh

set(TEST_TARGET bloom-1b4-zh)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# bloom-1b4-zh-quantize

set(TEST_TARGET bloom-1b4-zh-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
