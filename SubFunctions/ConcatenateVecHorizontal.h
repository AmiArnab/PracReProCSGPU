#ifndef ConcatenateVecHorizontal
#define ConcatenateVecHorizontal

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "thrust\device_new.h"
#include "thrust\device_delete.h"
#include "thrust\copy.h"
#include "thrust\find.h"
#include "thrust\device_ptr.h"

#include "ConcatenateVecHorizontal.cu"

#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int ConcatenateMatrix(float **dev_outmat, float *dev_inmat, float **dev_listmat, int M, int N, int n);

#endif
