cmake_minimum_required(VERSION 3.17)
project(NBody_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(NBody_CUDA main.cu simulation.cu)
target_link_libraries(NBody_CUDA sfml-graphics sfml-window sfml-system)

set_target_properties(
        NBody_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(NBody_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        >)

add_definitions(-DDRAW=1)