#
# mnist

set(TEST_TARGET mnist)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# mnist-cpu

set(TEST_TARGET mnist-cpu)
add_executable(${TEST_TARGET} main-cpu.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

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
        common
        common-ggml
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
    )
endif()
