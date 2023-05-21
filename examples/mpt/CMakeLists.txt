#
# mpt

set(TEST_TARGET mpt)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# mpt-quantize

set(TEST_TARGET mpt-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# mpt-perplexity

set(TEST_TARGET mpt-perplexity)
add_executable(${TEST_TARGET} perplexity.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
