#
# whisper

add_library(whisper-cpp
    whisper.cpp
    )

target_link_libraries(whisper-cpp PRIVATE
    ggml
    )

set(TEST_TARGET whisper)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE whisper-cpp common)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

#
# whisper-quantize

set(TEST_TARGET whisper-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
