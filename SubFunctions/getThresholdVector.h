#ifndef getThresholdVector
#define getThresholdVector

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "getThresholdVector.cu"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

int Thresh(unsigned int *T, float *dev_St, int M,float omega);

#endif
