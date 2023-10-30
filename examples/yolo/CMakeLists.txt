#
# yolov3-tiny

set(TEST_TARGET yolov3-tiny)
add_executable(${TEST_TARGET} yolov3-tiny.cpp yolo-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
