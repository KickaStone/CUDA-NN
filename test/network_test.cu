#include <gtest/gtest.h>
#include "../mnist_loader.h"
#include "../network.cuh"
#include "../dense.cuh"
#include "../conv.cuh"
#include "../maxpooling.cuh"


std::vector<double*> train_data, test_data;
std::vector<int> train_label, test_label;
TEST(Network, load_mnist) {
    char* train_image_file = "E:/Projects/Cuda/CUDA-NN/data/train-images.idx3-ubyte";
    const char* train_label_file = "E:/Projects/Cuda/CUDA-NN/data/train-labels.idx1-ubyte";
    const char* test_image_file = "E:/Projects/Cuda/CUDA-NN/data/t10k-images.idx3-ubyte";
    const char* test_label_file = "E:/Projects/Cuda/CUDA-NN/data/t10k-labels.idx1-ubyte";

    load_mnist(train_image_file, train_label_file, train_data, train_label);
    load_mnist(test_image_file, test_label_file, test_data, test_label);

    ASSERT_EQ(train_data.size(), 60000);
    ASSERT_EQ(train_label.size(), 60000);
    ASSERT_EQ(test_data.size(), 10000);
    ASSERT_EQ(test_label.size(), 10000);
}

// TEST(Network, create){
//     NeuralNetwork nn = NeuralNetwork(2, 784, 10);
//     nn.add_layer(new Dense(784, 30, CUDNN_ACTIVATION_SIGMOID));
//     nn.add_layer(new Dense(30, 10, CUDNN_ACTIVATION_SIGMOID));
//     nn.setData(train_data, train_label, test_data, test_label);
//     nn.setParams(10, 10, 3.0);
//     nn.train();
// }


TEST(Network, cnn){
    NeuralNetwork nn = NeuralNetwork(7, 784, 10);
    nn.add_layer(new ConvolutionalLayer(1, 28, 28, 6, 5, 1, 0, 1, CUDNN_ACTIVATION_RELU));
    nn.add_layer(new MaxPooling(6, 24, 24, 2, 2, 0, 1));
    nn.add_layer(new ConvolutionalLayer(6, 12, 12, 16, 5, 1, 0, 1, CUDNN_ACTIVATION_RELU));
    nn.add_layer(new MaxPooling(16, 8, 8, 2, 2, 0, 1));
    nn.add_layer(new Dense(4 * 4 * 16, 120, CUDNN_ACTIVATION_SIGMOID));
    nn.add_layer(new Dense(120, 84, CUDNN_ACTIVATION_SIGMOID));
    nn.add_layer(new Dense(84, 10, CUDNN_ACTIVATION_SIGMOID));
    nn.setData(train_data, train_label, test_data, test_label);
    nn.setParams(10, 10, 0.1);
    nn.train();
}