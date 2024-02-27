#
# gpt-2

set(TEST_TARGET gpt-2-ctx)
add_executable(${TEST_TARGET} main-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-alloc)
add_executable(${TEST_TARGET} main-alloc.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-backend)
add_executable(${TEST_TARGET} main-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

set(TEST_TARGET gpt-2-sched)
add_executable(${TEST_TARGET} main-sched.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-quantize

set(TEST_TARGET gpt-2-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# gpt-2-batched

set(TEST_TARGET gpt-2-batched)
add_executable(${TEST_TARGET} main-batched.cpp)
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
