#include "l1NormMinimization.h"

int performBregmanIterRegularization(float *dev_x, float *dev_yt, float *dev_Phit, float epsilon, int M, int N)
{
	float err = epsilon + 100;
	float mu = 1;
	float delta = 0.7;

	float *dev_temp1 = NULL;
	float *dev_currv = NULL;
	float *dev_u = NULL;
	float *dev_tempy = NULL;

	cublasHandle_t handle;
	cublasOperation_t opN = CUBLAS_OP_N;
	cublasOperation_t opT = CUBLAS_OP_T;

	float alpha1 = 1;
	float alpha2 = -1;
	float beta1 = 1;
	//float beta2 = -1;

	cudaMalloc((void **)&dev_u, M*sizeof(float));
	cudaMalloc((void **)&dev_currv, M*sizeof(float));
	cudaMalloc((void **)&dev_temp1, M*sizeof(float));
	cudaMalloc((void **)&dev_tempy, M*sizeof(float));

	thrust::device_vector<float> devyt(dev_yt, dev_yt + M - 1);
	thrust::device_vector<float> devx(dev_x, dev_x + M - 1);
	thrust::device_vector<float> devu(dev_u, dev_u + M - 1);
	thrust::device_vector<float> devcurrv(dev_currv, dev_currv + M - 1);
	thrust::device_vector<float> devtemp1(dev_temp1, dev_temp1 + M - 1);
	thrust::device_vector<float> devtempy(dev_tempy, dev_tempy + M - 1);
	

	while (err > epsilon)
	{
		//cublasSgemv_v2(handle, opN, M, M, &alpha2, dev_Phit, M, thrust::raw_pointer_cast(&devu[0]), 1, &beta1, thrust::raw_pointer_cast(&devtempy[0]), 1);
		cublasSgemv_v2(handle, opT, M, M, &alpha1, dev_Phit, M, thrust::raw_pointer_cast(&devtempy[0]), 1, &beta1, thrust::raw_pointer_cast(&devcurrv[0]), 1);

		for (int i = 0; i < M; ++i)
		{
			if (devcurrv[i] > mu)
			{
				devu[i] = delta*(devcurrv[i] - mu);
			}
			else if (devcurrv[i] < -mu)
			{
				devu[i] = delta*(devcurrv[i] + mu);
			}
			else
			{
				devu[i] = 0;
			}
		}

		thrust::copy(devyt.begin(), devyt.end(), devtempy.begin());
		cublasSgemv_v2(handle, opN, M, M, &alpha2, dev_Phit, M, thrust::raw_pointer_cast(&devu[0]), 1, &beta1, thrust::raw_pointer_cast(&devtempy[0]), 1);

		cublasSnrm2_v2(handle, M, thrust::raw_pointer_cast(&devtempy[0]), 1, &err);
	}

	cublasDestroy_v2(handle);

	cudaFree(dev_u);
	cudaFree(dev_tempy);
	cudaFree(dev_currv);
	cudaFree(dev_u);

	return 0;
}
