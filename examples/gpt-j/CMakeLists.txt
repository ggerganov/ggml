#
# gpt-j

set(TEST_TARGET gpt-j)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

if (MSVC)
    target_compile_features(${TEST_TARGET} PUBLIC cxx_std_20)
endif()

#
# gpt-j-quantize

set(TEST_TARGET gpt-j-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

if (MSVC)
    target_compile_features(${TEST_TARGET} PUBLIC cxx_std_20)
endif()