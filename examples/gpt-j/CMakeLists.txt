#
# gpt-j

set(TEST_TARGET gpt-j)
add_executable(${TEST_TARGET} gpt-j.cpp main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

#
# gpt-j-quantize

set(TEST_TARGET gpt-j-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
