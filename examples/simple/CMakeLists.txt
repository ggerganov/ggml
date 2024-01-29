#
# gpt-2

set(TEST_TARGET simple)
add_executable(${TEST_TARGET} simple.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
