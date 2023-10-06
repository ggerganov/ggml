#
# gpt-2

set(TEST_TARGET gpt-2)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# For GPU offloading

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

if (GGML_CLBLAST)
    add_compile_definitions(GGML_USE_CLBLAST)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
