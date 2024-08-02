#
# mnist-common

set(TEST_TARGET mnist-common)
add_library(${TEST_TARGET} mnist-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-fc-eval

set(TEST_TARGET mnist-fc-eval)
add_executable(${TEST_TARGET} mnist-fc-eval.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)

#
# mnist-fc-train

set(TEST_TARGET mnist-fc-train)
add_executable(${TEST_TARGET} mnist-fc-train.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)

#
# mnist-cnn

set(TEST_TARGET mnist-cnn)
add_executable(${TEST_TARGET} main-cnn.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-cpu

set(TEST_TARGET mnist-cpu)
add_executable(${TEST_TARGET} main-cpu.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (APPLE)
    #
    # mnist-mtl

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    set(TEST_TARGET mnist-mtl)
    add_executable(${TEST_TARGET} main-mtl.cpp main-mtl.h main-mtl.m)
    target_link_libraries(${TEST_TARGET} PRIVATE
        ggml
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
    )
endif()
