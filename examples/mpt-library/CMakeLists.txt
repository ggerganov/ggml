# OJ 2023 modded from ../mpt
# mpt-library

set(TEST_TARGET mpt-library)
add_library(${TEST_TARGET} mpt.h mpt_impl.cpp MPTWrapper.h MPTWrapper.cxx)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)