#ifndef getProjectionMatrix
#define getProjectionMatrix

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "getProjectionMatrix.cu"

#include <iostream>
#include <cstdio>

using namespace std;

int getProjectionMat(float *Phi, float *P, int M, int N);

#endif
