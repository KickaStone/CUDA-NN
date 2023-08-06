//
// Created by JunchengJi on 8/6/2023.
//

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../dense.cuh"
#include "../common.h"
#include <cmath>

//using namespace Eigen;
using namespace std;

Dense dense(3, 2, CUDNN_ACTIVATION_SIGMOID);

TEST(Dense, create){
    double *w = dense.get_weights();
    double *b = dense.get_bias();

    cublasPrintMat(w, 3, 2, "w");
    cublasPrintMat(b, 1, 2, "b");
}

TEST(Dense, forward){
    Eigen::MatrixXd w(3, 2);
    w << 0.1, 0.2,
         0.3, 0.4,
         0.5, 0.6;
    Eigen::MatrixXd b(2, 1);
    b << 0.1, 0.2;
    dense.set_weights(w.data());
    dense.set_bias(b.data());


    Eigen::MatrixXd input(3, 1);
    input << 0.1, 0.2, 0.3;

    Eigen::MatrixXd r = w.transpose() * input + b;
    cout << r.unaryExpr([](double a){return 1/(1+exp(-a));}) << endl;
    CUDA_CHECK(cudaSetDevice(0));
    double *d_input;
    CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(double) * 3));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), sizeof(double) * 3, cudaMemcpyHostToDevice));

    double *output = dense.forward(d_input);
    cublasPrintMat(output, 2, 1, "output");
}

TEST(Dense, backward){
    Eigen::MatrixXd output_grad(2, 1);
    output_grad << 0.5, 0.6;
    double *d_output_grad;
    CUDA_CHECK(cudaMalloc((void**)&d_output_grad, sizeof(double) * 2));
    CUDA_CHECK(cudaMemcpy(d_output_grad, output_grad.data(), sizeof(double) * 2, cudaMemcpyHostToDevice));
    
    dense.backward(d_output_grad);
    double *dw = dense.get_d_weights();
    double *db = dense.get_d_bias();

    cublasPrintMat(dw, 3, 2, "dw");
    cublasPrintMat(db, 2, 1, "db");
}

TEST(Dense, udpate){
    dense.update(0.5);
    cublasPrintMat(dense.get_weights(), 3, 2, "w");
    cublasPrintMat(dense.get_bias(), 1, 2, "b");
}