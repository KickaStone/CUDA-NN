#include "layer.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

class Dense : public Layer
{
public:
    Dense(int input_size, int output_size, cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_SIGMOID);
    ~Dense();

    double* forward(double *input) override;
    double* backward(double *input) override;
    void update(double lr) override;

    [[nodiscard]]double* get_weights() const { return weights; }
    [[nodiscard]]double* get_bias() const { return bias; }
    [[nodiscard]]double* get_d_weights() const { return d_weights; }
    [[nodiscard]]double* get_d_bias() const { return d_bias; }
    void set_weights(double *w) { cublasSetMatrix(input_size, output_size, sizeof(double), w, input_size, weights, input_size); }
    void set_bias(double *b) { cublasSetMatrix(output_size, 1, sizeof(double), b, output_size, bias, output_size); }

private:
    cublasHandle_t cublas_handle{};
    cudnnHandle_t cudnn_handle{};

    double *input{};
    double *a{};
    double *z{};
    double *dz{};
    double *weights{}; // weights is a matrix of size (output_size, input_size), but int cublas view is (input_size, output_size)
    double *bias{};
    double *d_weights{};
    double *d_bias{};
    double *input_grad{};

    cudnnActivationMode_t activation_mode{CUDNN_ACTIVATION_IDENTITY}; // if no activation, use identity
    cudnnActivationDescriptor_t act_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t weights_desc;
};