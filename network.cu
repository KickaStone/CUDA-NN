#include "network.cuh"

__global__ void cal_l2_grad(double *output, int y, double *grad, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        if(i == y) {
            grad[i] = output[i] - 1;
        } else {
            grad[i] = output[i];
        }
    }
}

NeuralNetwork::NeuralNetwork(int n, int i, int o) : num_layers(n), num_input(i), num_output(o) {

}

NeuralNetwork::~NeuralNetwork() {
    for(int i = 0; i < num_layers; i++) {
        delete layers[i];
    }
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

void NeuralNetwork::train()
{   
    if(is_valid() == false) {
        throw std::runtime_error("invalid network");
    }
    
    double *grad;
    double *h_grad;
    CUDA_CHECK(cudaMalloc(&grad, sizeof(double) * num_output));
    CUDA_CHECK(cudaMallocHost(&h_grad, sizeof(double) * num_output));
    for(int e = 0; e < epoch; e++) {
        double loss = 0.0;
        for(int i = 0; i < train_data.size(); i += batch_size){
            for(int j = 0; j < batch_size; j++) {
                double *output = forward(train_data[i + j]);
                cal_l2_grad<<<(num_output + 255) / 256, 256>>>(output, train_labels[i + j], grad, num_output);
                cublasGetMatrix(num_output, 1, sizeof(double), grad, num_output, h_grad, num_output);
                for(int k = 0; k < num_output; k++) {
                    loss += h_grad[k] * h_grad[k];
                }
                backward(grad);
            }
            update(lr/batch_size);
        }
        std::cout << "epoch: " << e << " loss: " << loss/2 << ", acc: " << predict() << "/" << test_data.size() << std::endl;
    }
    CUDA_CHECK(cudaFree(grad));
    CUDA_CHECK(cudaFreeHost(h_grad));
}

double* NeuralNetwork::forward(double *x) {
    double *input = x;
    for(int i = 0; i < num_layers; i++) {
        input = layers[i]->forward(input);
        // cublasPrintMat(input, 1, layers[i]->get_output_size(), "input_" + std::to_string(i));
    }
    return input;
}

double* NeuralNetwork::backward(double *gard){
    double *grad = gard;
    for(int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
        // cublasPrintMat(grad, 1, layers[i]->get_input_size(), "grad_" + std::to_string(i));
    }
    return grad;
}

void NeuralNetwork::setData(std::vector<double *> &train_data, std::vector<int> &train_labels, std::vector<double *> &test_data, std::vector<int> &test_labels)
{
    // this->train_data = train_data;
    for(int i = 0 ; i < train_data.size(); i++){
        double *tmp;
        CUDA_CHECK(cudaMalloc((void **)&tmp, sizeof(double) * num_input));
        CUDA_CHECK(cudaMemcpy(tmp, train_data[i], sizeof(double) * num_input, cudaMemcpyHostToDevice));
        this->train_data.push_back(tmp);
    }

    this->train_labels = train_labels;

    // this->test_data = test_data;
    for(int i = 0 ; i < test_data.size(); i++){
        double *tmp;
        CUDA_CHECK(cudaMalloc((void **)&tmp, sizeof(double) * num_input));
        CUDA_CHECK(cudaMemcpy(tmp, test_data[i], sizeof(double) * num_input, cudaMemcpyHostToDevice));
        this->test_data.push_back(tmp);
    }
    this->test_labels = test_labels;
}

bool NeuralNetwork::is_valid()
{
    // check layer
    if(num_layers != layers.size())
        throw std::runtime_error("num_layers != layers.size()");
    int input = num_input;
    for(int i = 0; i < num_layers; i++) {
        if(layers[i] == nullptr)
            throw std::runtime_error("layers[" + std::to_string(i) + "] == nullptr");

        if(layers[i]->get_input_size() != input)
            throw std::runtime_error("layers[" + std::to_string(i) + "]->get_input_size() != last output");
        input = layers[i]->get_output_size();
    }
    if(input != num_output)
        throw std::runtime_error("input != num_output");

    // check data
    if(train_data.size() == 0 || train_data.size() != train_labels.size())
        throw std::runtime_error("train_data invalid");

    if(test_data.size() == 0 || test_data.size() != test_labels.size())
        throw std::runtime_error("test_data invalid");
    
    // check params
    if(epoch <= 0 || batch_size <= 0 || lr <= 0)
        throw std::runtime_error("params invalid");

    return true;
}

void NeuralNetwork::setParams(int epoch, int batch_size, double lr)
{
    this->epoch = epoch;
    this->batch_size = batch_size;
    this->lr = lr;
}

void NeuralNetwork::update(double lr) {
    for(int i = 0; i < num_layers; i++) {
        layers[i]->update(lr);
    }
}


int NeuralNetwork::predict(){
    Eigen::MatrixXd h_output(num_output, 1);
    int correct = 0;
    for(int i = 0; i < test_data.size(); i++){
        double *output = forward(test_data[i]);
        cublasGetMatrix(num_output, 1, sizeof(double), output, num_output, h_output.data(), num_output);
        int tmp_i, tmp_j;
        h_output.maxCoeff(&tmp_i, &tmp_j);
        if(tmp_i == test_labels[i]) {
            correct++;
        }
    }
    return correct;
}
