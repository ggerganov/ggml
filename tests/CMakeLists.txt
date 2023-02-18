# on APPLE - include Accelerate framework
if (APPLE AND NOT GGML_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

#
# test-vec0

set(TEST_TARGET test-vec0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-vec1 (x86)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86")
    set(TEST_TARGET test-vec1)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_FLAGS "-mavx -mavx2 -mfma -mf16c")
endif()

#
# test-vec2 (arm)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(TEST_TARGET test-vec2)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

#
# test-grad0

set(TEST_TARGET test-grad0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-mul-mat0

set(TEST_TARGET test-mul-mat0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-mul-mat1 (arm)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND NOT GGML_NO_ACCELERATE)
    set(TEST_TARGET test-mul-mat1)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

#
# test-mul-mat2

set(TEST_TARGET test-mul-mat2)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test0

set(TEST_TARGET test0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test1

set(TEST_TARGET test1)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test2

set(TEST_TARGET test2)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test3

set(TEST_TARGET test3)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-svd0 (arm)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND NOT GGML_NO_ACCELERATE)
    set(TEST_TARGET test-svd0)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

