#
# whisper

set(TEST_TARGET whisper)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
