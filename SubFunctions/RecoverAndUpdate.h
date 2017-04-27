#ifndef RecoverAndUpdate
#define RecoverAndUpdate

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "RecoverAndUpdate.cu"
//#include "getApproxBasisUpdate.h"

#include <iostream>
#include <cstdlib>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int UpdateSubspace(thrust::device_vector<float> *devPt, float *dev_Lt, thrust::device_vector<float> *devPLt, int M, int N, int d, int rank, bool dosvd);

#endif
