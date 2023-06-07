#
# gpt-neox

set(TEST_TARGET gpt-neox)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-neox-quantize

set(TEST_TARGET gpt-neox-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
