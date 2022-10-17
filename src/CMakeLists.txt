if (GGML_ALL_WARNINGS)
    if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
        #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wshadow                        \
            -Wcast-qual                     \
            -Wstrict-prototypes             \
            -Wpointer-arith                 \
        ")
    else()
        # todo : windows
    endif()
endif()

# compiler flags

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=vla")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno -ffinite-math-only -funsafe-math-optimizations")

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-m1")
else()
    message(STATUS "x86 detected")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -mfma -mf16c")
endif()


# ggml

set(TARGET ggml)

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

if (GGML_PERF)
    set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_PERF)
endif()

add_library(${TARGET}
    ggml.c
    )

target_include_directories(${TARGET} PUBLIC
    .
    ../include
    ../include/ggml
    )

target_link_libraries(${TARGET} PUBLIC m ${GGML_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET} PUBLIC
        ${CMAKE_DL_LIBS}
        )

    target_compile_definitions(${TARGET} PUBLIC
        GGML_SHARED
        )
endif()

target_compile_definitions(${TARGET} PUBLIC
    ${GGML_EXTRA_FLAGS}
    )

if (MINGW)
    target_link_libraries(${TARGET} PUBLIC
        stdc++
        )
endif()

install(TARGETS ${TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    )
