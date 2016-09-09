#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include <cstdio>
using namespace std;

int getProjectionMatrix(float *Phi, float *P, int M, int N)
{
	float *dev_Phi = NULL;
	float *dev_P = NULL;
	float *dev_I = NULL;
	float *host_I = NULL;
	float *host_P = NULL;
	float *host_Phi = NULL;
	float alpha = 1;
	float beta = 0;
	float gamma = -1;

	cublasHandle_t chandle;
	cublasOperation_t topA = CUBLAS_OP_N;
	cublasOperation_t topB = CUBLAS_OP_T;

	cudaError cerr;

	host_I = new float[M*M];
	host_P = new float[M*N];
	host_Phi = new float[M*M];

	cerr = cudaMalloc((void **)&dev_Phi, M*M*sizeof(float));
	cerr = cudaMalloc((void **)&dev_P, M*N*sizeof(float));
	cerr = cudaMalloc((void **)&dev_I, M*M*sizeof(float));

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			host_I[IDX2C(i,j,M)] = (i==j)?1:0;
		}
	}

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			cout << host_I[IDX2C(i, j, M)] << " ";
		}
		cout << endl;
	}

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			host_P[IDX2C(i, j, M)] = P[(i*M) + j];
		}
	}

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			cout << host_P[IDX2C(i, j, M)] << " ";
		}
		cout << endl;
	}

	cublasCreate_v2(&chandle);

	cublasSetMatrix(M, N, sizeof(float), P, M, dev_P, M);
	cublasSetMatrix(M, M, sizeof(float), Phi, M, dev_Phi, M);
	cublasSetMatrix(M, M, sizeof(float), host_I, M, dev_I, M);

	cublasSgemm(chandle, topA, topB, M, M, N,&alpha,dev_P,M,dev_P,M,&beta,dev_Phi,M);
	cublasSgeam(chandle, topA, topA, M, N, &alpha, dev_I, M, &gamma, dev_Phi, M, dev_Phi, M);

	cublasGetMatrix(M, M, sizeof(float), dev_Phi, M, host_Phi, M);

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			Phi[(i*M) + j] = host_Phi[IDX2C(i, j, M)];
		}
	}

	cudaFree(dev_Phi);
	cudaFree(dev_P);

	delete[] host_I;
	delete[] host_P;
	delete[] host_Phi;

	return 0;
}

int main()
{
	int M = 3;
	int N = 3;
	float *arr = NULL;
	float *phi = NULL;

	arr = new float[M*N];
	phi = new float[M*M];

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			arr[(i*M)+j] = (i==j)?1:0;
		}
	}

	getProjectionMatrix(phi, arr, M, N);

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			cout << phi[(i*M) + j] << " ";
		}
		cout << endl;
	}

	getchar();
	delete[] arr;
    return 0;
}