// custom_cuda_layer.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_elementwise_add(float* input1, float* input2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}
