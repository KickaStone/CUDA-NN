cmake_minimum_required(VERSION 3.26)
set(this cuda_nn)
project(${this} CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(
        "c:/Program Files/NVIDIA/CUDNN/v8.9/include"
        ../cudnn-frontend/include
)

link_directories(
        "c:/Program Files/NVIDIA/CUDNN/v8.9/lib/x64"
)

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
)

find_package(Eigen3 REQUIRED)

include(GoogleTest)
target_link_libraries(
        ${this} PUBLIC
        cudnn
        Eigen3::Eigen
        cublas
        curand
        GTest::gtest
        GTest::gtest_main
)