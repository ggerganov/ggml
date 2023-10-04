cmake_minimum_required (VERSION 3.3)
project(ggml VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
    include(cmake/GitVars.cmake)
    include(cmake/BuildTypes.cmake)
else()
    set(GGML_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# options

option(BUILD_SHARED_LIBS            "ggml: build shared libs" ${BUILD_SHARED_LIBS_DEFAULT})

option(GGML_ALL_WARNINGS            "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY  "ggml: enable all compiler warnings in 3rd party libs" OFF)

option(GGML_SANITIZE_THREAD         "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS        "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED      "ggml: enable undefined sanitizer" OFF)

option(GGML_BUILD_TESTS             "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES          "ggml: build examples" ${GGML_STANDALONE})

option(GGML_TEST_COVERAGE           "ggml: enable test coverage" OFF)

option(GGML_PERF                    "ggml: enable perf timings"          OFF)
option(GGML_NO_ACCELERATE           "ggml: disable Accelerate framework" OFF)
option(GGML_OPENBLAS                "ggml: use OpenBLAS"                 OFF)
option(GGML_CLBLAST                 "ggml: use clBLAST"                  OFF)
option(GGML_CUBLAS                  "ggml: use cuBLAS"                   OFF)
option(GGML_METAL                   "ggml: use Metal"                    OFF)

# sanitizers

if (GGML_SANITIZE_THREAD)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

if (GGML_SANITIZE_ADDRESS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if (GGML_SANITIZE_UNDEFINED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

# instruction set specific
option(GGML_AVX                     "ggml: enable AVX"                                     ON)
option(GGML_AVX2                    "ggml: enable AVX2"                                    ON)
option(GGML_AVX512                  "ggml: enable AVX512"                                  OFF)
option(GGML_AVX512_VBMI             "ggml: enable AVX512-VBMI"                             OFF)
option(GGML_AVX512_VNNI             "ggml: enable AVX512-VNNI"                             OFF)
option(GGML_FMA                     "ggml: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(GGML_F16C                "ggml: enable F16C"                                    ON)
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=native")

# warning flags

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags   -Wall -Wpedantic -Wformat=2 -Wno-unused -Wstrict-prototypes)
        set(cxx_flags -Wall -Wpedantic -Wformat=2)
    else()
        # todo : windows
    endif()

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )
endif()

if (NOT MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:-Werror=vla>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Werror=vla>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-Werror=vla>"
    )
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity
# are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "DragonFly")
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WHISPER_PERF)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_PERF)
endif()

# dependencies

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)

# main

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

if (GGML_BUILD_TESTS)
    if (GGML_TEST_COVERAGE)
        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fprofile-instr-generate -fcoverage-mapping")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        else()
            message(WARNING "Test coverage is only supported for Clang")
        endif()
    endif()
endif()

add_subdirectory(src)

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
               @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
