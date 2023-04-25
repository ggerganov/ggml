#
# whisper

add_library(whisper-cpp
    whisper.cpp
    )

target_link_libraries(whisper-cpp PRIVATE
    ggml
    )

if (MSVC)
    target_compile_features(whisper-cpp PUBLIC cxx_std_20)
endif()

set(TEST_TARGET whisper)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE whisper-cpp common)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if (MSVC)
    target_compile_features(${TEST_TARGET} PUBLIC cxx_std_20)
endif()

#
# whisper-quantize

set(TEST_TARGET whisper-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

if (MSVC)
    target_compile_features(${TEST_TARGET} PUBLIC cxx_std_20)
endif()