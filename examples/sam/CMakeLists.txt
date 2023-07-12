#
# sam

set(TEST_TARGET sam)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# sam-quantize

#set(TEST_TARGET sam-quantize)
#add_executable(${TEST_TARGET} quantize.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
