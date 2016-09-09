
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cusolverDn.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

cudaError_t cudaStatus;

void printError(float pos)
{
	if (cudaStatus != cudaSuccess)
	{
		cout << "Error! " << pos << "\nEnter any key to continue...";
		getchar();
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) cout << "Device reset not successful!\nExiting...\n";
		exit(EXIT_FAILURE);
	}
	else
	{
		cout << "Success! " << pos << endl;
	}
}

void ExitProgram()
{
	cudaStatus = cudaDeviceReset();
	printError(-1);
	exit(EXIT_FAILURE);
}

int main()
{
	cusolverDnHandle_t my_handle;
	cusolverDnCreate(&my_handle);

	const int M = 3;
	const int N = 3;
	float *host_A = NULL;
	float *host_S = NULL;

	float *dev_A = NULL;
	float *dev_S = NULL;
	float *dev_U = NULL;
	float *dev_V = NULL;

	float *work = 0;
	int work_size = 0;
	int *dev_info = 0;

	int rank = 0;
	float threshold = 0.001;

	cout << "Starting program...\n";

	host_A = new float[M*N*sizeof(float)];
	host_S = new float[M*sizeof(float)];

	for (unsigned int i = 0; i < M; ++i)
	{
		for (unsigned int j = 0; j < N; ++j)
		{
			*(host_A + (i*M) + j) = i*j + 1;
		}
	}

	for (unsigned int i = 0; i < M; ++i)
	{
		for (unsigned int j = 0; j < N; ++j)
		{
			cout << *(host_A + (i*M) + j) << " ";
		}
		cout << endl;
	}

	cout << "Host allocation and initialization finished!\nStarting GPU allocation and copy...\n";

	cudaStatus = cudaSetDevice(0);
	printError(0);

	cudaStatus = cudaMalloc((void**)&dev_A, M * N * sizeof(float));
	printError(1.1);
	cudaStatus = cudaMalloc((void**)&dev_S, M * sizeof(float));
	printError(1.2);
	cudaStatus = cudaMalloc((void**)&dev_U, M * M * sizeof(float));
	printError(1.3);
	cudaStatus = cudaMalloc((void**)&dev_V, N * N * sizeof(float));
	printError(1.4);

	cudaStatus = cudaMemcpy(dev_A, host_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	printError(2);

	cusolverStatus_t cusolverstatus;

	cusolverstatus = cusolverDnSgesvd_bufferSize(my_handle, M, N, &work_size);
	if (cusolverstatus != CUSOLVER_STATUS_SUCCESS) { cout << "cuSolver buffer allocation failed!\n";  cudaDeviceReset(); exit(EXIT_FAILURE);}

	cudaStatus = cudaMalloc((void**)&work, work_size * sizeof(float));
	printError(3.1);
	cudaStatus = cudaMalloc((void**)&dev_info, sizeof(int));
	printError(3.2);

	cusolverstatus = cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_A, M, dev_S, dev_U, M, dev_V, N, work, work_size, NULL, dev_info);
	cudaStatus = cudaDeviceSynchronize();
	if (cusolverstatus != CUSOLVER_STATUS_SUCCESS) { cout << "SVD failed!\n"; cudaDeviceReset(); exit(EXIT_FAILURE);}
	printError(4.1);

	cout << "SVD completed!\n";

	cudaStatus = cudaMemcpy(host_S,dev_S,M*sizeof(float),cudaMemcpyDeviceToHost);
	printError(4.2);

	cudaStatus = cudaFree(dev_A);
	printError(5.1);
	cudaStatus = cudaFree(dev_S);
	printError(5.2);
	cudaStatus = cudaFree(dev_U);
	printError(5.3);

    cudaStatus = cudaDeviceReset();
	printError(6);

	for (unsigned int i = 0; i < M; ++i)
	{
		if (*(host_S + i) > threshold) ++rank;
	}

	cout << "Singular value threshold for computing rank: " << threshold << endl;
	cout << "Rank: " << rank << endl;

	delete [] host_A;
	delete [] host_S;

	cout << "Enter anything to exit...\n";
	getchar();

    return 0;
}
