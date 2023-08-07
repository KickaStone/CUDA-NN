#include <iostream>
#include <string>
#include <vector>
#include "layer.cuh"

struct Params{
    int epoch;
    int batch_size;
    double lr;
    double decay_rate;
};
class NeuralNetwork{
    public:
        NeuralNetwork(int n, int i, int o);
        ~NeuralNetwork();

        void add_layer(Layer* layer);
        void train();
        void update(double lr);
        double* forward(double *x);
        double* backward(double *grad);
        int predict();
        void setParams(int epoch, int batch_size, double lr, double decay_rate);
        void setData(std::vector<double*> &train_data, std::vector<int> &train_labels, std::vector<double*> &test_data, std::vector<int> &test_labels);

    private:
        int num_layers;
        int num_input;
        int num_output;
        Params params;
        std::vector<Layer*> layers;    
        std::vector<double*> train_data;
        std::vector<int> train_labels;
        std::vector<double*> test_data;
        std::vector<int> test_labels;

        bool is_valid();
        void shuffle();
        double decay_lr(int epoch);
};