#include <gtest/gtest.h>

#include <cudnn.h>
#include "../common.h"
#include <cublas_v2.h>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

TEST(cublas, MatMul){
    Matrix3d A, B;
    A << 1,2,3,
         4,5,6,
         7,8,9;
    B << 1,2,3,
        4,5,6,
        7,8,9;

    cout << "A:" << endl << A << endl;
    cout << "B:" << endl << B << endl;
    
    Matrix3d C = A * B;
    cout << C << endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * 9);
    cudaMalloc(&d_B, sizeof(double) * 9);
    cudaMalloc(&d_C, sizeof(double) * 9);

    cudaMemcpy(d_A, A.data(), sizeof(double) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(double) * 9, cudaMemcpyHostToDevice);

    cublasSetMatrix(3, 3, sizeof(double), A.data(), 3, d_A, 3);
    cublasSetMatrix(3, 3, sizeof(double), B.data(), 3, d_B, 3);
    cublasSetMatrix(3, 3, sizeof(double), C.data(), 3, d_C, 3);

    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &alpha, d_A, 3, d_B, 3, &beta, d_C, 3);

    cublasGetMatrix(3, 3, sizeof(double), d_C, 3, C.data(), 3);

    cout << C << endl;
    CUBLAS_CHECK(cublasDestroy(handle));

}
