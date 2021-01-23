cmake_minimum_required(VERSION 3.17)
project(NBody_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(NBody_CUDA main.cu)

set_target_properties(
        NBody_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)