#ifndef getMatrixRank
#define getMatrixRank

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\copy.h"
#include "thrust\device_ptr.h"
#include "thrust\sort.h"
#include "thrust\device_new.h"
#include "thrust\device_delete.h"
#include "thrust\iterator\counting_iterator.h"

#include "getMatrixRank.cu"

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

//int getMatRank(int *matrank, float *dev_Mt, float threshold, int M, int N);
int getMatRank(int *matrank, float *dev_Mt, float threshold, int M, int N);

#endif
