cmake_minimum_required (VERSION 3.0)
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

# options

option(GGML_ALL_WARNINGS            "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY  "ggml: enable all compiler warnings in 3rd party libs" OFF)

option(GGML_SANITIZE_THREAD         "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS        "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED      "ggml: enable undefined sanitizer" OFF)

option(GGML_BUILD_TESTS             "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES          "ggml: build examples" ${GGML_STANDALONE})

option(GGML_PERF                    "ggml: enable perf timings"          OFF)
option(GGML_NO_ACCELERATE           "ggml: disable Accelerate framework" OFF)
option(GGML_OPENBLAS                "ggml: use OpenBLAS"                 OFF)
option(GGML_CLBLAST                 "ggml: use clBLAST"                  OFF)
option(GGML_CUBLAS                  "ggml: use cuBLAS"                   OFF)

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

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=native")

# dependencies

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)

# main

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

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
