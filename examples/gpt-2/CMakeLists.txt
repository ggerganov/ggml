#
# gpt-2

set(TEST_TARGET gpt-2)
add_executable(${TEST_TARGET} gpt-2.cpp main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
