project(nntest)

add_executable(nntest 
    # cublas_test.cu
    # dense_test.cu
    cudnn_test.cu
    # conv_test.cu
    maxpooling_test.cu
    network_test.cu
)

find_package(Eigen3 REQUIRED)

target_link_libraries(nntest
    PUBLIC
    cuda_nn
)

include(GoogleTest)

gtest_discover_tests(nntest)