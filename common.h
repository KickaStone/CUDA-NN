#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <iostream>
#include <cublas_v2.h>
#include <Eigen/Dense>

#define CUDA_CHECK(expression)                               \
  {                                                          \
    cudaError_t status = (expression);                       \
    if (status != cudaSuccess) {                             \
      std::cerr << __FILE__  << " Error on line " << __LINE__ << ": "      \
                << cudaGetErrorString(status) << std::endl;  \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define CUDNN_CHECK(expression)                              \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr <<  __FILE__ << " Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#define CUBLAS_CHECK(expression)                             \
  {                                                          \
    cublasStatus_t status = (expression);                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << __FILE__ << " Error on line " << __LINE__ << ": "      \
                << cublasGetStatusString(status) << std::endl;\
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


static void cublasPrintMat(double* mat, int m, int n, std::string tag){
    std::cout << tag << std::endl;
    Eigen::MatrixXd A(m, n);
    cublasGetMatrix(m, n, sizeof(double), mat, m, A.data(), m);
    std::cout << A << std::endl;
}

#endif //CUDA_CHECK_H

