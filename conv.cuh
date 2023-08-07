#ifndef _CONVOLUTIONAL_LAYER_H_
#define _CONVOLUTIONAL_LAYER_H_

#include "layer.cuh"

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int channel_in, int height_in, int width_in, int channel_out, 
    int kernel_size, int dilation, int padding, int stride, cudnnActivationMode_t activation_mode);
    ~ConvolutionalLayer();

    double* forward(double* x) override;
    double* backward(double* output_grad) override;
    void update(double learning_rate) override;

    void set_kernel(int i, double *kernel) {
        cudaMemcpy(K + i * kernel_size * kernel_size * channel_in, kernel, 
        kernel_size * kernel_size * channel_in * sizeof(double), cudaMemcpyHostToDevice);
        // cublasPrintMat(K + i * kernel_size * kernel_size * channel_in, kernel_size, kernel_size, "K " + std::to_string(i));
    }

    [[nodiscard]]double* get_kernel(int i){return K + i * kernel_size * kernel_size * channel_in;}
    [[nodiscard]]double* get_bias(int i){return b + i * height_out * width_out;}
    [[nodiscard]]double* get_activation(int i){ return a + i * height_out * width_out;}

    [[nodiscard]]double* get_input(){return input;}
    [[nodiscard]]double* get_input_grad(){return input_grad;}
    [[nodiscard]]double* get_dK(){return dK;}
    [[nodiscard]]double* get_db(){return db;}
    [[nodiscard]]double* get_da(){return da;}

    // [[nodiscard]]int get_channel_in(){return channel_in;}
    // [[nodiscard]]int get_height_in(){return height_in;}
    // [[nodiscard]]int get_width_in(){return width_in;}
    // [[nodiscard]]int get_channel_out(){return channel_out;}
    // [[nodiscard]]int get_kernel_size(){return kernel_size;}
    // [[nodiscard]]int get_dilation(){return dilation;}
    // [[nodiscard]]int get_padding(){return padding;}
    // [[nodiscard]]int get_stride(){return stride;}
    // [[nodiscard]]int get_height_out(){return height_out;}
    // [[nodiscard]]int get_width_out(){return width_out;}
    // [[nodiscard]]cudnnActivationMode_t get_activation_mode(){return activation_mode;}

private:
    int channel_in{};
    int height_in{};
    int width_in{};
    int channel_out{};
    int kernel_size{};
    int dilation{};
    int padding{};
    int stride{};
    int height_out{};
    int width_out{};

    double *K{}; // kernels
    double *b{}; // bias
    double *a{}; // activation

    double *input{};
    double *input_grad{};
    double *dK{};
    double *db{};
    double *da{};
    double *dz{};

    cudnnHandle_t cudnn_handle;
    cudnnActivationMode_t activation_mode{CUDNN_ACTIVATION_RELU};
    cudnnActivationDescriptor_t act_desc_identity;
    cudnnActivationDescriptor_t act_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t dK_desc;
};

#endif // _CONVOLUTIONAL_LAYER_H_
