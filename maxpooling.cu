#include "maxpooling.cuh"

MaxPooling::MaxPooling(int channel_in, int height_in, int width_in, int kernel_size, int stride, int pad, int dilation)
    : channel_in(channel_in), height_in(height_in), width_in(width_in),
      kernel_h(kernel_size), kernel_w(kernel_size), stride_h(stride), stride_w(stride), pad_h(pad), pad_w(pad), dilation_h(dilation), dilation_w(dilation)
{
    channel_out = channel_in;
    height_out = (height_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    width_out = (width_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;

    this->input_size = channel_in * height_in * width_in;
    this->output_size = channel_out * height_out * width_out;

    CUDNN_CHECK(cudnnCreate(&cudnnHandle));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            kernel_h,
                                            kernel_w,
                                            pad_h,
                                            pad_w,
                                            stride_h,
                                            stride_w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_DOUBLE,
                                           1,
                                           channel_in,
                                           height_in,
                                           width_in));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_DOUBLE,
                                           1,
                                           channel_out,
                                           height_out,
                                           width_out));
    int n;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc,
                                                  input_desc,
                                                  &n,
                                                  &channel_out,
                                                  &height_out,
                                                  &width_out));
    this->input_size = channel_in * height_in * width_in;
    this->output_size = channel_out * height_out * width_out;
    // allocate a
    CUDA_CHECK(cudaMalloc((void **)&a, sizeof(double)*output_size));
    CUDA_CHECK(cudaMalloc((void **)&input_grad, sizeof(double)*input_size));
}

MaxPooling::~MaxPooling()
{
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroy(cudnnHandle));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(input_grad));
}

double *MaxPooling::forward(double *input)
{
    double alpha = 1.0;
    double beta = 0.0;
    CUDNN_CHECK(cudnnPoolingForward(cudnnHandle,
                                    pooling_desc,
                                    &alpha,
                                    input_desc,
                                    input,
                                    &beta,
                                    output_desc,
                                    a));
    return a;
}

double *MaxPooling::backward(double *grad_input)
{
    double alpha = 1.0;
    double beta = 0.0;
    CUDNN_CHECK(cudnnPoolingBackward(cudnnHandle,
                                     pooling_desc,
                                     &alpha,
                                     output_desc,
                                     a,
                                     output_desc,
                                     grad_input,
                                     input_desc,
                                     a,
                                     &beta,
                                     input_desc,
                                     input_grad));
    return input_grad;
}

void MaxPooling::update(double lr)
{
    return;
}
