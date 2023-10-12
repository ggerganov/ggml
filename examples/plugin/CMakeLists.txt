add_library(plugin-model STATIC model.cpp)
target_link_libraries(plugin-model PUBLIC ggml::ggml)

add_executable(cpu-plugin cpu-plugin.cpp)
target_link_libraries(cpu-plugin plugin-model)

if (GGML_CUBLAS)
    add_executable(cuda-plugin cuda-plugin.cpp)
    target_link_libraries(cuda-plugin plugin-model)
endif()
