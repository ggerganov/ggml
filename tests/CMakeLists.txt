# check systems
if (NOT UNAME_S)
    execute_process(COMMAND uname -s OUTPUT_VARIABLE UNAME_S)
endif()
if (NOT UNAME_P)
    execute_process(COMMAND uname -p OUTPUT_VARIABLE UNAME_P)
endif()
if (NOT UNAME_M)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME_M)
endif()
#message(STATUS "UNAME_S: ${UNAME_S}  UNAME_P: ${UNAME_P}  UNAME_M: ${UNAME_M}")

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
if (UNAME_S MATCHES "Darwin")
    if (NOT UNAME_P MATCHES "arm")
        execute_process(COMMAND sysctl -n hw.optional.arm64 OUTPUT_VARIABLE SYSCTL_M)
        if (SYSCTL_M MATCHES "1")
            #set(UNAME_P "arm")
            #set(UNAME_M "arm64")
            message(WARNING "Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lea
d to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-#1282546789")
        endif()
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    #set(GGML_C_FLAGS "${GGML_C_FLAGS} -mcpu=apple-m1")
else()
    message(STATUS "x86 detected")
    #set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx -mavx2 -mfma -mf16c")
    if (UNAME_S MATCHES "Darwin")
        execute_process(COMMAND sysctl machdep.cpu.features OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "AVX1.0")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND sysctl machdep.cpu.leaf7_features OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "AVX2")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx2")
        endif()
        if (AVX1_M MATCHES "FMA")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mfma")
        endif()
        set(GGML_C_FLAGS "${GGML_C_FLAGS} -mf16c")
    elseif (UNAME_S MATCHES "Linux")
        message(STATUS "Linux detected")
        execute_process(COMMAND grep "avx " /proc/cpuinfo OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND grep "avx2 " /proc/cpuinfo OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx2")
        endif()
        execute_process(COMMAND grep "fma " /proc/cpuinfo OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mfma")
        endif()
        execute_process(COMMAND grep "f16c " /proc/cpuinfo OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mf16c")
        endif()
        execute_process(COMMAND grep "sse3 " /proc/cpuinfo OUTPUT_VARIABLE SSE3_M)
        if (SSE3_M MATCHES "sse3")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -msse3")
        endif()
    elseif (UNAME_S MATCHES "Haiku")
        message(STATUS "Haiku detected")
	execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX " OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX2 " OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mavx2")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "FMA " OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mfma")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "F16C " OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(GGML_C_FLAGS "${GGML_C_FLAGS} -mf16c")
        endif()
    else()
        set(GGML_C_FLAGS  "${GGML_C_FLAGS} -mfma -mf16c -mavx -mavx2")
    endif()
endif()

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

if (GGML_OPENBLAS)
    set(OPENBLAS_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/include/openblas
        /usr/include/openblas-base
        /usr/local/include
        /usr/local/include/openblas
        /usr/local/include/openblas-base
        /opt/OpenBLAS/include
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/include
        )
    find_path(OPENBLAS_INC NAMES cblas.h PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
    find_library(OPENBLAS_LIB NAMES openblas libopenblas)
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  ${OPENBLAS_LIB})
        set(GGML_EXTRA_INCS  ${GGML_EXTRA_INCS}  ${OPENBLAS_INC})
        set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

#
# test-vec0

set(TEST_TARGET test-vec0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# test-vec1 (x86)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86")
    set(TEST_TARGET test-vec1)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    #set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_FLAGS "-mavx -mavx2 -mfma -mf16c")
    set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_FLAGS ${GGML_C_FLAGS})
endif()

#
# test-vec2 (arm)
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(TEST_TARGET test-vec2)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
endif()

#
# test-grad0

set(TEST_TARGET test-grad0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-opt

set(TEST_TARGET test-opt)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-quantize-fns

set(TEST_TARGET test-quantize-fns)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-quantize-perf

set(TEST_TARGET test-quantize-perf)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-mul-mat0

set(TEST_TARGET test-mul-mat0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-mul-mat1 (arm)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND NOT GGML_NO_ACCELERATE)
    set(TEST_TARGET test-mul-mat1)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
endif()

#
# test-blas0 (arm)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND NOT GGML_NO_ACCELERATE)
    set(TEST_TARGET test-blas0)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> 128 128 128)
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
if (MSVC)
    target_link_options(${TEST_TARGET} PRIVATE "/STACK: 8388608") # 8MB
endif()
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
# test-pool

set(TEST_TARGET test-pool)
add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

#
# test-svd0 (arm/x86)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" AND NOT GGML_NO_ACCELERATE)
    set(TEST_TARGET test-svd0)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86" AND GGML_OPENBLAS)
    set(TEST_TARGET test-svd0)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${GGML_EXTRA_LIBS})
    target_compile_options(${TEST_TARGET} PRIVATE ${GGML_EXTRA_FLAGS})
endif()

