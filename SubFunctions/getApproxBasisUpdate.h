#ifndef getApproxBasisUpdate
#define getApproxBasisUpdate

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "cublas_v2.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "getApproxBasisUpdate.cu"

#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

int GetApproxBasisUpdate(thrust::device_vector<float> *appbase, float *dev_Mtrain, int M, int N, int r);

#endif
