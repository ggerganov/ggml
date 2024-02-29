#
# yolov3-tiny

set(TEST_TARGET superpoint)
add_executable(${TEST_TARGET} run_superpoint.cpp superpoint-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
