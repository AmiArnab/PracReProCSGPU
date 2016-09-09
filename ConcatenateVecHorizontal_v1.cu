#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

int getMatrixRank(float *inmat,int M,int N,int *rank,float errthreshold)
{
	cusolverDnHandle_t my_handle;
	cusolverDnCreate(&my_handle);

	float *host_A = inmat;
	float *host_S = NULL;

	float *dev_A = NULL;
	float *dev_S = NULL;
	float *dev_U = NULL;
	float *dev_V = NULL;

	float *work = 0;
	int work_size = 0;
	int *dev_info = 0;

	host_S = new float[M*sizeof(float)];

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_A, M * N * sizeof(float));
	cudaMalloc((void**)&dev_S, M * sizeof(float));
	cudaMalloc((void**)&dev_U, M * M * sizeof(float));
	cudaMalloc((void**)&dev_V, N * N * sizeof(float));
	cudaMemcpy(dev_A, host_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	cusolverDnSgesvd_bufferSize(my_handle, M, N, &work_size);
	cudaMalloc((void**)&work, work_size * sizeof(float));
	cudaMalloc((void**)&dev_info, sizeof(int));

	cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_A, M, dev_S, dev_U, M, dev_V, N, work, work_size, NULL, dev_info);
	cudaDeviceSynchronize();

	cudaMemcpy(host_S, dev_S, M*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_A);
	cudaFree(dev_S);
	cudaFree(dev_U);

	for (unsigned int i = 0; i < M; ++i)
	{
		if (*(host_S + i) > errthreshold) ++(*rank);
	}

	delete[] host_S;
	return 0;
}

int main()
{
	const int M = 3;
	const int N = 3;

	int rank = 0;

	cout << "Staring program...\n";

	float *host_A = NULL;
	host_A = new float[M*N*sizeof(float)];

	for (unsigned int i = 0; i < M; ++i)
	{
		for (unsigned int j = 0; j < N; ++j)
		{
			*(host_A + (i*M) + j) = i*j + 1;
		}
	}

	getMatrixRank(host_A, M, N, &rank, 0.001);

	cout << "Rank : " << rank << endl;

	getchar();
	cudaDeviceReset();

    return 0;
}