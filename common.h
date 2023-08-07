#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <iostream>
#include <cublas_v2.h>

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
    double *h_tmp;

    CUDA_CHECK(cudaMallocHost((void **)&h_tmp, m * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(h_tmp, mat, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    for(int i = 0; i < m; i ++){
      for(int j = 0; j < n; j++){
        std::cout << h_tmp[i * n + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    CUDA_CHECK(cudaFreeHost(h_tmp));
}

#endif //CUDA_CHECK_H

