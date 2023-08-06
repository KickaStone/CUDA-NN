#ifndef _LAYER_CUH_
#define _LAYER_CUH_
#include "common.h"
#include <cudnn.h>
#include <curand.h>


class Layer{
    protected:
    int input_size;
    int output_size;

    public:
    [[nodiscard]] int get_input_size() const { return input_size; }
    [[nodiscard]] int get_output_size() const { return output_size; }
    virtual double* forward(double *input) = 0;
    virtual double* backward(double *input) = 0;
    virtual void update(double coeff) = 0;
};

#endif // _LAYER_CUH_