#ifndef l1NormMinimization
#define l1NormMinimization

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"
#include "thrust\device_ptr.h"

#include <cstdio>
#include <cstdlib>

#include "l1NormMinimization.cu"

//Include thrust to use C++ STL

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

int performBregmanIterRegularization(float *dev_x, float *dev_yt, float *dev_Phit, float epsilon, int M, int N);

#endif
