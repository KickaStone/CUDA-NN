#ifndef _MAXPOOLING_H_
#define _MAXPOOLING_H_

#include "layer.cuh"

class MaxPooling : public Layer
{
public:
    MaxPooling(int channel_in,
                int height_in,
                int width_in,
                int kernel_size,
                int stride,
                int pad,
                int dilation                
    );
    ~MaxPooling();
    double* forward(double *input) override;
    double* backward(double *grad_input) override;
    void update(double lr) override;

private:
    int channel_in{};
    int height_in{};
    int width_in{};
    int kernel_h{};
    int kernel_w{};
    
    int stride_h{};
    int stride_w{};
    int pad_h{};
    int pad_w{};
    int dilation_h{};
    int dilation_w{};

    int channel_out{};
    int height_out{};
    int width_out{};


    double *input;
    double *a;
    double *input_grad;
    
    cudnnHandle_t cudnnHandle;
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
};


#endif // _MAXPOOLING_H_