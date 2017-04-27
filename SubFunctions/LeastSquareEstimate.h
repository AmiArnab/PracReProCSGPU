#ifndef LeastSquareEstimate
#define LeastSquareEstimate

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "cublas_v2.h"

#include "LeastSquareEstimate.cu"

#include <iostream>
#include <cstdio>
#include <cstdlib>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int getLeastSquareEstimate(float *Stadd, float *yt, float *Phit, unsigned int *Tt, int M, int N);

#endif
