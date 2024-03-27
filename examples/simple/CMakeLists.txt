#
# simple-ctx

set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# simple-backend

set(TEST_TARGET simple-backend)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
