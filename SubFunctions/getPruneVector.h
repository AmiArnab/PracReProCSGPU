#ifndef getPruneVector
#define getPruneVector

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "getPruneVector.cu"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

int Prune(unsigned int *T, float *St, int M, unsigned int s);

#endif
