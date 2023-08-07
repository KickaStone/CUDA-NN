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
    double learning_rate = 0;
    CUDA_CHECK(cudaMalloc(&grad, sizeof(double) * num_output));
    CUDA_CHECK(cudaMallocHost(&h_grad, sizeof(double) * num_output));
    for(int e = 0; e < params.epoch; e++) {
        shuffle();
        double loss = 0.0;
        learning_rate = decay_lr(e);
        for(int i = 0; i < train_data.size(); i += params.batch_size){
            for(int j = 0; j < params.batch_size; j++) {
                double *output = forward(train_data[i + j]);
                cal_l2_grad<<<(num_output + 255) / 256, 256>>>(output, train_labels[i + j], grad, num_output);
                cublasGetMatrix(num_output, 1, sizeof(double), grad, num_output, h_grad, num_output);
                for(int k = 0; k < num_output; k++) {
                    loss += h_grad[k] * h_grad[k];
                }
                backward(grad);
            }
            update(learning_rate / params.batch_size);
        }
        std::cout << "epoch: " << e <<  "/" << params.epoch << "\tlr: " << 
            learning_rate << "\tloss: " << loss/(2 * train_data.size()) << "\tacc: " << 
            predict() << "/" << test_data.size() << std::endl;
    }
    CUDA_CHECK(cudaFree(grad));
    CUDA_CHECK(cudaFreeHost(h_grad));
}

double* NeuralNetwork::forward(double *x) {
    double *input = x;
    for(int i = 0; i < num_layers; i++) {
        input = layers[i]->forward(input);
    }
    return input;
}

double* NeuralNetwork::backward(double *gard){
    double *grad = gard;
    for(int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
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
    if(params.epoch <= 0 || params.batch_size <= 0 || params.lr <= 0)
        throw std::runtime_error("params invalid");

    return true;
}

void NeuralNetwork::shuffle()
{
    srand(time(NULL));
    for(int i = 0; i < train_data.size(); i++){
        int j = rand() % train_data.size();
        std::swap(train_data[i], train_data[j]);
        std::swap(train_labels[i], train_labels[j]);
    }
}

double NeuralNetwork::decay_lr(int epoch)
{
    return params.lr / (1 + epoch * params.decay_rate);
}

void NeuralNetwork::setParams(int epoch, int batch_size, double lr, double decay_rate)
{
    params.epoch = epoch;
    params.batch_size = batch_size;
    params.lr = lr;
    params.decay_rate = decay_rate;
}

void NeuralNetwork::update(double lr) {
    for(int i = 0; i < num_layers; i++) {
        layers[i]->update(lr);
    }
}

int NeuralNetwork::predict(){
    double *h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, sizeof(double) * num_output));
    int correct = 0;
    int max_i = 0;
    for(int i = 0; i < test_data.size(); i++){
        double *output = forward(test_data[i]);
        cublasGetMatrix(num_output, 1, sizeof(double), output, num_output, h_output, num_output);
        max_i = 0;
        for(int j = 0; j < num_output; j++) {
            if(h_output[j] > h_output[max_i]) {
                max_i = j;
            }
        }
        if(max_i == test_labels[i]) {
            correct++;
        }
    }
    CUDA_CHECK(cudaFreeHost(h_output));
    return correct;
}
