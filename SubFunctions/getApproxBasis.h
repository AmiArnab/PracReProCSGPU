#ifndef getApproxBasis
#define getApproxBasis

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "getApproxBasis.cu"

#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

//int GetApproxBasis(float *dev_approx_basis, int *Ma, int *Na, float *dev_Mtrain, int M, int N, float b);
int GetApproxBasis(float *dev_approx_basis, int *Ma, int *Na, float *dev_Mtrain, int M, int N, float b);

#endif
