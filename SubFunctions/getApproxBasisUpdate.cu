#include "getApproxBasisUpdate.h"
int GetApproxBasisUpdate(thrust::device_vector<float> *appbase, float *dev_Mtrain, int M, int N, int r)
{
	cusolverDnHandle_t my_handle;
	cusolverDnCreate(&my_handle);

	float *dev_S = NULL;
	float *dev_U = NULL;
	float *dev_V = NULL;

	float *work = 0;
	int work_size = 0;
	int *dev_info = 0;

	thrust::device_vector<float> devU(M*M);

	cudaMalloc((void**)&dev_S, M * sizeof(float));
	cudaMalloc((void**)&dev_U, M * M * sizeof(float));
	cudaMalloc((void**)&dev_V, N * N * sizeof(float));

	cusolverDnSgesvd_bufferSize(my_handle, M, N, &work_size);
	cudaMalloc((void**)&work, work_size * sizeof(float));
	cudaMalloc((void**)&dev_info, sizeof(int));
	cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_Mtrain, M, dev_S, dev_U, M, dev_V, N, work, work_size, NULL, dev_info);
	cudaDeviceSynchronize();
	
	float const alpha(1.0);
	float const beta(0.0);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, M, &alpha, dev_U, M, &beta, NULL, M, thrust::raw_pointer_cast(&devU[0]), M);
	cublasDestroy(handle);
	cudaDeviceSynchronize();

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < r; ++j)
		{
			(*appbase)[i*r + j] = devU[i*M + j];
		}
	}


	cudaFree(dev_V);
	cudaFree(dev_S);
	cudaFree(dev_U);

	cusolverDnDestroy(my_handle);

	return 0;
}
