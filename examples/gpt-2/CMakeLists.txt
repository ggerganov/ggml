#
# gpt-2

set(TEST_TARGET gpt-2)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
