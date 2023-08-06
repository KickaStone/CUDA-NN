#include "conv.cuh"

ConvolutionalLayer::ConvolutionalLayer(int channel_in, int height_in, int width_in, int channel_out, int kernel_size, int dilation, int padding, int stride, cudnnActivationMode_t activation_mode)
: channel_in(channel_in), height_in(height_in), width_in(width_in), channel_out(channel_out), kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), activation_mode(activation_mode)
{
    this->input_size = channel_in * height_in * width_in;

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dK_desc));

    int t;
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channel_in, height_in, width_in));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, channel_out, channel_in, kernel_size, kernel_size));
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &t, &channel_out, &height_out, &width_out));

    this->output_size = channel_out * height_out * width_out;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channel_out, height_out, width_out));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channel_out, height_out, height_out));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dK_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, channel_out, channel_in, kernel_size, kernel_size));
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, activation_mode, CUDNN_PROPAGATE_NAN, 0.0));

    CUDA_CHECK(cudaMalloc((void**)&K, sizeof(double) * channel_out * channel_in * kernel_size * kernel_size));
    CUDA_CHECK(cudaMalloc((void**)&b, sizeof(double) * channel_out * height_out * width_out));
    CUDA_CHECK(cudaMalloc((void**)&a, sizeof(double) * channel_out * height_out * width_out));
    CUDA_CHECK(cudaMalloc((void**)&input_grad, sizeof(double) * channel_in * height_in * width_in));
    CUDA_CHECK(cudaMalloc((void**)&dK, sizeof(double) * channel_out * channel_in * kernel_size * kernel_size));
    CUDA_CHECK(cudaMalloc((void**)&db, sizeof(double) * channel_out * height_out * width_out));

    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateNormalDouble(generator, K, channel_out * channel_in * kernel_size * kernel_size, 0.0, 1.0);
    curandGenerateNormalDouble(generator, b, channel_out * height_out * width_out, 0.0, 1.0);
    
    // CUDA_CHECK(cudaMemset(b, 0, sizeof(double) * channel_out * height_out * width_out));
    CUDA_CHECK(cudaMemset(a, 0, sizeof(double) * channel_out * height_out * width_out));


    // init backward
    CUDA_CHECK(cudaMemset(input_grad, 0, sizeof(double) * channel_in * height_in * width_in));
    CUDA_CHECK(cudaMemset(dK, 0, sizeof(double) * channel_out * channel_in * kernel_size * kernel_size));
    CUDA_CHECK(cudaMemset(db, 0, sizeof(double) * channel_out * height_out * width_out));

    curandDestroyGenerator(generator);
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    CUDNN_CHECK(cudnnDestroy(cudnn_handle));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dK_desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));

    CUDA_CHECK(cudaFree(K));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));

    // CUDA_CHECK(cudaFree(input_grad));
}

double *ConvolutionalLayer::forward(double *x)
{
    input = x;
    double alpha = 1.0, beta = 0.0;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input, filter_desc, K, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, a));
    CUDNN_CHECK(cudnnAddTensor(cudnn_handle, &alpha, bias_desc, b, &alpha, output_desc, a)); 
    CUDNN_CHECK(cudnnActivationForward(cudnn_handle, act_desc, &alpha, output_desc, a, &beta, output_desc, a));
    return a;
}

double *ConvolutionalLayer::backward(double *output_grad)
{
    double *d_dz;
    CUDA_CHECK(cudaMalloc((void**)&d_dz, sizeof(double) * channel_out * height_out * width_out));
    double alpha = 1.0, beta = 0.0;
    CUDNN_CHECK(cudnnActivationBackward(cudnn_handle, act_desc, &alpha, output_desc, a, output_desc, output_grad, output_desc, a, &beta, output_desc, d_dz));
    
    // for(int i = 0; i < channel_out; i++){
    //     cublasPrintMat(d_dz + i * height_out * width_out, height_out, width_out, "d_dz" + std::to_string(i) + ": ");
    // }
    
    double *db2;
    CUDA_CHECK(cudaMalloc((void**)&db2, sizeof(double) * channel_out));
    cudnnTensorDescriptor_t db2_desc;
    CUDA_CHECK(cudaMemset(db2, 0, sizeof(double) * channel_out));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&db2_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(db2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channel_out, 1, 1));
    CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle, &alpha, output_desc, d_dz, &alpha, db2_desc, db2));

    // CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle, &alpha, output_desc, d_dz, &alpha, bias_desc, db));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, input_desc, input, output_desc, d_dz, conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, nullptr, 0, &alpha, filter_desc, dK));
    CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, K, output_desc, d_dz, conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, nullptr, 0, &beta, input_desc, input_grad));
    return input_grad;
}

void ConvolutionalLayer::update(double learning_rate)
{
    double alpha = -learning_rate;
    double beta = 1.0;
    CUDNN_CHECK(cudnnAddTensor(cudnn_handle, &alpha, bias_desc, db, &alpha, bias_desc, b));
    CUDNN_CHECK(cudnnAddTensor(cudnn_handle, &alpha, dK_desc, dK, &beta, dK_desc, K));
    // reset gradient
    CUDA_CHECK(cudaMemset(db, 0, sizeof(double) * channel_out * height_out * width_out));
    CUDA_CHECK(cudaMemset(dK, 0, sizeof(double) * channel_out * channel_in * kernel_size * kernel_size));
}
