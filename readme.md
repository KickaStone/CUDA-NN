- [Cuda Implementation of CNN](#cuda-implementation-of-cnn)
  - [Dependency](#dependency)
  - [Layers](#layers)
  - [Model Overview](#model-overview)
  - [Params](#params)
  - [result](#result)

# Cuda Implementation of CNN

## Dependency
* CUDA Toolkit 12.2 
* cuBlas -- Matrix Operation
* cuDNN v8.9 -- Convolution forward and backward, activation 

## Layers
* Conv2d: 28x28x1  6filters 5x5 s=1 p=2 RELU
* MaxPooling2d: 28x28x6 s=2
* Conv2d: 14x14x6 16filters 5x5 s=1 p=0 RELU
* MaxPooling2d: 10x10x16 s=2
* Dense: 160 SIGMOID
* Dense: 84  SIGMOID
* Dense: 10  SIGMOID

## Model Overview

![overview](/img/overview.png)

## Params
* Epoch 30
* BatchSize 10
* LearningRate 0.1
* DelayRate 0

## result
Accurate : 98.2%

Running time: 1036.790s

