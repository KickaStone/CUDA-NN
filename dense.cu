#include "dense.cuh"


Dense::Dense(int input_size, int output_size, cudnnActivationMode_t mode)
{
    CUDA_CHECK(cudaMalloc((void **)&weights, input_size * output_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&bias, output_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&a, output_size * sizeof(double)));
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation_mode = mode;

    curandGenerator_t curand_generator;
    curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_generator, CURAND_RNG_PSEUDO_MT19937);
    curandGenerateNormalDouble(curand_generator, weights, input_size * output_size, 0, 0.1);
    curandGenerateNormalDouble(curand_generator, bias, output_size, 0, 0.1);

    CUDA_CHECK(cudaMalloc((void **)&d_weights, input_size * output_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_bias, output_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&input_grad, input_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_weights, 0.0, input_size * output_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bias, 0.0, output_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(input_grad, 0.0, input_size * sizeof(double)));

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));

    curandDestroyGenerator(curand_generator);
}

Dense::~Dense()
{
    try{
        CUDA_CHECK(cudaFree(weights));
        CUDA_CHECK(cudaFree(bias));
        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_bias));
        CUDA_CHECK(cudaFree(input_grad));

        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        CUDNN_CHECK(cudnnDestroy(cudnn_handle));
    }catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }
}

double* Dense::forward(double *input_data) {
    this->input = input_data;
    double alpha = 1.0;
    double beta = 0.0;
    CUDA_CHECK(cudaMemset(a, 0.0, output_size * sizeof(double)));

    // a = weights * input
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             output_size, 1, input_size,
                             &alpha, weights, input_size, input_data, input_size,
                             &beta, a, output_size));
//    cublasPrintMat(bias, output_size, 1);
//    cublasPrintMat(a, output_size, 1);


    // a += bias
    CUBLAS_CHECK(cublasDaxpy(cublas_handle, output_size, &alpha, bias, 1, a, 1));

//    cublasPrintMat(a, output_size, 1, "a");

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, output_size, 1));

    cudnnActivationDescriptor_t act_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
    CUDNN_CHECK(cudnnActivationForward(cudnn_handle, act_desc, &alpha, desc, a, &beta, desc, a));

    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroyActivationDescriptor(act_desc);
    return a;
}

double* Dense::backward(double *output_grad) {

    cudnnActivationDescriptor_t act_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
    
    cudnnTensorDescriptor_t a_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&a_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(a_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, output_size, 1));

    cudnnTensorDescriptor_t d_a_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&d_a_desc));

    double *d_dz;
    CUDA_CHECK(cudaMalloc((void **)&d_dz, output_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dz, 0.0, output_size * sizeof(double)));

    double alpha = 1.0;
    double beta = 0.0;
    CUDNN_CHECK(cudnnActivationBackward(cudnn_handle, act_desc, &alpha, a_desc, a, a_desc, output_grad, a_desc, input, &beta, a_desc, d_dz));
    
    // std::cout << "d_dz" << std::endl;
    // cublasPrintMat(d_dz, output_size, 1);
    
    // calculate d_weights
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             input_size, output_size, 1,
                             &alpha, input, input_size, d_dz, 1,
                             &alpha, d_weights, input_size));

    // std::cout << "d_weights" << std::endl;
    // cublasPrintMat(d_weights, input_size, output_size);

    // calculate d_bias
    CUBLAS_CHECK(cublasDaxpy(cublas_handle, output_size, &alpha, d_dz, 1, d_bias, 1));
    // cublasPrintMat(d_bias, output_size, 1, "d_bias");


    // calculate d_input
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             input_size, 1, output_size,
                             &alpha, weights, input_size, d_dz, output_size,
                             &beta, input_grad, input_size));
    // std::cout << "d_input" << std::endl;
    // cublasPrintMat(input_grad, input_size, 1);

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    return input_grad;
}

void Dense::update(double lr) {
    // update weights and bias
    double alpha = -lr;
    CUBLAS_CHECK(cublasDaxpy(cublas_handle, input_size * output_size, &alpha, d_weights, 1, weights, 1));
    CUBLAS_CHECK(cublasDaxpy(cublas_handle, output_size, &alpha, d_bias, 1, bias, 1));

    // reset d_weights and d_bias
    CUDA_CHECK(cudaMemset(d_weights, 0.0, input_size * output_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bias, 0.0, output_size * sizeof(double)));
}




