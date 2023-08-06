#include <gtest/gtest.h>
#include "../conv.cuh"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

ConvolutionalLayer *conv;

TEST(Conv, create){
    conv = new
    ConvolutionalLayer (
        1, 3, 3, 2, 2, 1, 0, 1,
        CUDNN_ACTIVATION_RELU
    );
}

TEST(Conv, forward){
    Matrix3d input;
    input << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    vector<Matrix2d> kernels;
    kernels.push_back(Matrix2d::Ones(2, 2));
    kernels.push_back(Matrix2d::Identity(2, 2));
    
    conv->set_kernel(0, kernels[0].data());
    conv->set_kernel(1, kernels[1].data());
    
    for(int i = 0; i < 2; i++){
        auto t = conv->get_kernel(i);
        cublasPrintMat(t, 2, 2, "kernel" + std::to_string(i));

        auto b = conv->get_bias(i);
        cublasPrintMat(b, 2, 2, "bias" + std::to_string(i));
    }


    double *d_input;
    cudaMalloc(&d_input, sizeof(double) * 3 * 3);
    cublasSetMatrix(3, 3, sizeof(double), input.data(), 3, d_input, 3);
    auto output = conv->forward(d_input);    
    cublasPrintMat(output, 2, 2, "output");
    cublasPrintMat(output+4, 2, 2, "output2");
}


TEST(conv, backward){
    double grad[] = {1, 2, 3, 4, 4, 3, 2, 1};
    double *d_output_grad;
    cudaMalloc(&d_output_grad, sizeof(double) * 2 * 2 * 2);
    cudaMemcpy(d_output_grad, grad, sizeof(double) * 2 * 2 * 2, cudaMemcpyHostToDevice);
    auto input_grad = conv->backward(d_output_grad);
    cublasPrintMat(input_grad, 3, 3, "input_grad");
}

