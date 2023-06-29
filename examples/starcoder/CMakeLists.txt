#
# starcoder

set(TEST_TARGET starcoder)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# starcoder-mmap

set(TEST_TARGET starcoder-mmap)
add_executable(${TEST_TARGET} starcoder-mmap.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# starcoder-quantize

set(TEST_TARGET starcoder-quantize)
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

