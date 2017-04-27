#ifndef GetSupportSet
#define GetSupportSet

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "thrust\device_vector.h"
#include "thrust\device_ptr.h"
#include "thrust\device_allocator.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"

#include "GetSupportSet.cu"

#include <iostream>
#include <cstdio>

using namespace std;

int getSupportSet(int *supportset, float *inputarr, int N);
float findIntersection(int *Told1, int *Told2, int M);
float getSetDifference(int *Told1, int *Told2, int M);
int getSupportCardinality(int *T, int M);
float computeOmega(float *dev_Mt, int M);
int computeLt(float *dev_Lt, float *dev_Mt, float *dev_St, int M);

#endif
