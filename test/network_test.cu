#include <gtest/gtest.h>
#include "../mnist_loader.h"
#include "../network.cuh"
#include "../dense.cuh"


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

TEST(Network, create){
    NeuralNetwork nn = NeuralNetwork(2, 784, 10);
    nn.add_layer(new Dense(784, 30));
    nn.add_layer(new Dense(30, 10));
    nn.setData(train_data, train_label, test_data, test_label);
    nn.setParams(10, 10, 3.0);
    nn.train();
}


