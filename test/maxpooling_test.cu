#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../common.h"
#include "../maxpooling.cuh"
#include <cmath>

using namespace Eigen;
using namespace std;

MaxPooling *pooling;
TEST(maxpooling, creat){
    pooling = new MaxPooling(1, 8, 8, 2, 2, 0, 1);
}