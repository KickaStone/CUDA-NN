#include <gtest/gtest.h>
#include <cudnn.h>
#include <Eigen/Dense>
#include <iostream>
#include <cublas_v2.h>


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