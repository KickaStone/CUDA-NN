cmake_minimum_required(VERSION 3.26)
set(this cuda_nn)
project(${this} CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(test)

add_library(${this} STATIC
        layer.cuh
        dense.cuh
        dense.cu
        network.cuh
        network.cu
        mnist_loader.h
        conv.cuh
        conv.cu
        maxpooling.cuh
        maxpooling.cu
)

include(GoogleTest)
target_link_libraries(
        ${this} PUBLIC
        cudnn
        cublas
        curand
        GTest::gtest
        GTest::gtest_main
)
