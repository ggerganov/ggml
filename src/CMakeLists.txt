# ggml

add_library(ggml ggml.c ggml-alloc.c)

set_property(TARGET ggml PROPERTY PUBLIC_HEADER
     ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml/ggml.h
     ${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml/ggml-alloc.h)

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        add_compile_options(-Wunused -Wextra -Wcast-qual -Wdouble-promotion)
        add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wshadow;-Wno-unused-function;-Wmissing-prototypes>")
    else()
        # todo : windows
    endif()
endif()

# compiler flags

if (NOT MSVC)
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno -ffinite-math-only -funsafe-math-optimizations")
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

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
	    message(WARNING "Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-#1282546789")
	endif()
    endif()
endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten")
    message(STATUS "Emscripten detected")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-m1")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PPC64 detected")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mpower9-vector")
else()
    message(STATUS "x86 detected")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -mfma -mf16c")
    if (UNAME_S MATCHES "Darwin")
        execute_process(COMMAND sysctl machdep.cpu.features OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "AVX1.0")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND sysctl machdep.cpu.leaf7_features OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "AVX2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        if (AVX1_M MATCHES "FMA")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
    elseif (UNAME_S MATCHES "Linux")
        message(STATUS "Linux detected")
        execute_process(COMMAND grep "avx " /proc/cpuinfo OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND grep "avx2 " /proc/cpuinfo OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        execute_process(COMMAND grep "fma " /proc/cpuinfo OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        execute_process(COMMAND grep "f16c " /proc/cpuinfo OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
        execute_process(COMMAND grep "sse3 " /proc/cpuinfo OUTPUT_VARIABLE SSE3_M)
        if (SSE3_M MATCHES "sse3")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
        endif()
    elseif (UNAME_S MATCHES "Haiku")
        message(STATUS "Haiku detected")
        execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX " OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX2 " OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        execute_process(COMMAND sysinfo -cpu COMMAND grep "FMA " OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        execute_process(COMMAND sysinfo -cpu COMMAND grep "F16C " OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
    elseif (MSVC)
        if (GGML_AVX512)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (GGML_AVX512_VBMI)
                add_compile_definitions(__AVX512VBMI__)
            endif()
            if (GGML_AVX512_VNNI)
                add_compile_definitions(__AVX512VNNI__)
            endif()
        elseif (GGML_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        elseif (GGML_AVX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX")
        endif()
    else()
        set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -mfma -mf16c -mavx -mavx2")
    endif()
endif()

# on APPLE - include Accelerate framework
if (APPLE AND NOT GGML_NO_ACCELERATE)
    ggml_wrap_library(accelerate::accelerate ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  accelerate::accelerate)
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
    ggml_wrap_library(openblas::openblas OPENBLAS_LIB NAMES openblas libopenblas)
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS} openblas::openblas)
        set_target_properties(openblas::openblas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INC})
	set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (GGML_CLBLAST)
	set(CLBLAST_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/local/include
	    $ENV{CLBLAST_HOME}
	    $ENV{CLBLAST_HOME}/include
        )
	find_path(CLBLAST_INC NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS})
	ggml_wrap_library(opencl::opencl OPENCL_LIB NAMES OpenCL)
	ggml_wrap_library(clblast::clblast CLBLAST_LIB NAMES clblast)
	if (CLBLAST_LIB AND OPENCL_LIB AND CLBLAST_INC)
		message(STATUS "clBLAST found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS} clblast::clblast opencl::opencl)
        set_target_properties(clblast::clblast PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CLBLAST_INC})
		set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_CLBLAST)

        target_sources(ggml PRIVATE ggml-opencl.cpp)
        set_property(TARGET ggml APPEND PROPERTY PUBLIC_HEADER ggml-opencl.h)
    else()
        message(WARNING "clBLAST not found")
    endif()
endif()

if (GGML_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    ggml_find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        target_sources(ggml PRIVATE ggml-cuda.cu)
        set_property(TARGET ggml APPEND PROPERTY PUBLIC_HEADER ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (GGML_STATIC)
            set(GGML_EXTRA_LIBS ${GGML_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(GGML_EXTRA_LIBS ${GGML_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()
    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (GGML_METAL)
    ggml_wrap_library(foundation::foundation            FOUNDATION_LIBRARY         Foundation              REQUIRED)
    ggml_wrap_library(metal::metal                      METAL_FRAMEWORK            Metal                   REQUIRED)
    ggml_wrap_library(metalkit::metalkit                METALKIT_FRAMEWORK         MetalKit                REQUIRED)
    ggml_wrap_library(metalperformace::metalperformance METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    target_sources(ggml PRIVATE ggml-metal.m)
    set_property(TARGET ggml APPEND PROPERTY PUBLIC_HEADER ggml-metal.h)

    add_compile_definitions(GGML_USE_METAL)
    add_compile_definitions(GGML_METAL_NDEBUG)

    # get full path to the file
    #add_compile_definitions(GGML_METAL_DIR_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/")

    # copy ggml-metal.metal to bin directory
    configure_file(ggml-metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)

    set(GGML_EXTRA_LIBS ${GGML_EXTRA_LIBS}
        foundation::foundation
        metal::metal
        metalkit::metalkit
        metalperformace::metalperformance
        )
endif()

if (GGML_PERF)
    set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_PERF)
endif()


set_target_properties(ggml PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    INTERFACE_GGML_MAJOR_VERSION ${PROJECT_VERSION_MAJOR}
    )

set_property(TARGET ggml APPEND PROPERTY
    COMPATIBLE_INTERFACE_STRING GGML_MAJOR_VERSION
    )

target_include_directories(ggml PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/ggml>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    ${GGML_EXTRA_INCS}
    )

if (MSVC)
    target_link_libraries(ggml PUBLIC ${GGML_EXTRA_LIBS} Threads::Threads)
else()
    target_link_libraries(ggml PUBLIC m ${GGML_EXTRA_LIBS} Threads::Threads)
endif()

if (BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

    target_link_libraries(ggml PUBLIC
        ${CMAKE_DL_LIBS}
        )

    target_compile_definitions(ggml PUBLIC
        GGML_SHARED
        )

    target_compile_definitions(ggml PRIVATE
        GGML_BUILD
        )

    if (GGML_METAL)
        set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")
    endif()
endif()

target_compile_definitions(ggml PUBLIC
    ${GGML_EXTRA_FLAGS}
    )

if (MINGW)
    target_link_libraries(ggml PUBLIC stdc++)
endif()

if (GGML_CUDA_SOURCES)
    message(STATUS "GGML CUDA sources found, configuring CUDA architecture")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES "52;61")
    set_property(TARGET ggml PROPERTY CUDA_SELECT_NVCC_ARCH_FLAGS "Auto")
    if (NOT MSVC)
        target_link_libraries(ggml PUBLIC stdc++)
    endif()
endif()

install(TARGETS ggml
    EXPORT GGMLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/static
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ggml
    )
