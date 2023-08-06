#include <gtest/gtest.h>
#include <cudnn.h>
#include <Eigen/Dense>
#include <iostream>
#include <cublas_v2.h>

#include "../common.h"


using namespace Eigen;
using namespace std;

TEST(cudnn, actBack){

    VectorXd a(3);
    a << 0.1, 0.2, 0.3;

    cout << a.unaryExpr([](double x){return x * (1-x);}) << endl;
    double *d_a;
    

    cudaMalloc((void**)&d_a, 3 * sizeof(double));

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, 1, 3);
    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
    double alpha = 1.0;
    double beta = 0.0;
    cublasSetMatrix(8, 1, sizeof(double), a.data(), 8, d_a, 8);


    cudnnActivationBackward(handle, actDesc, &alpha, desc, d_a, desc, d_a, desc, d_a, &beta, desc, d_a);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cublasGetMatrix(8, 1, sizeof(double), d_a, 8, a.data(), 8);
    cout << a << endl;
    
    cudnnDestroyTensorDescriptor(desc);

}

TEST(cudnn, conv){
    
        MatrixXd a(3, 3);
        a << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

        MatrixXd b(2, 2);
        b << 1, 2,
            3, 4;

        double *d_a;
        double *d_b;
        double *d_c;

        CUDA_CHECK(cudaMalloc((void**)&d_a, 9 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_b, 4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_c, 4 * sizeof(double)));

        CUBLAS_CHECK(cublasSetMatrix(9, 1, sizeof(double), a.data(), 9, d_a, 9));
        CUBLAS_CHECK(cublasSetMatrix(4, 1, sizeof(double), b.data(), 4, d_b, 4));

        cudnnHandle_t handle;
        CUDNN_CHECK(cudnnCreate(&handle));

        cudnnTensorDescriptor_t descA;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&descA));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(descA, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, 3, 3));

        cudnnFilterDescriptor_t descB;
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&descB));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(descB, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, 1, 2, 2));

        cudnnConvolutionDescriptor_t descConv;
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&descConv));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(descConv, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));


        
}