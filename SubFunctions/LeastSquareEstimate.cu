#include "LeastSquareEstimate.h"

int getLeastSquareEstimate(float *Stadd, float *yt, float *Phit, unsigned int *Tt, int M, int N)
{
	float *dev_yt = NULL;
	float *dev_Phit = NULL;
	float *dev_tempA = NULL;

	float *dev_auxArray = NULL;
	const float *dev_Array[1] = {dev_auxArray}; //Change this
	float *dev_ArrayLU[1] = { dev_auxArray };
	int *dev_pivotarrr = NULL;
	int *dev_infoarr = NULL;
	float *dev_Carray[1]; //Change this

	//cudaError_t error;
	cublasHandle_t bhandle;
	cusolverDnHandle_t shandle;
	cusolverDnCreate(&shandle);
	//cusolverStatus_t status;
	cublasOperation_t opN = CUBLAS_OP_N;
	cublasOperation_t opT = CUBLAS_OP_T;
	float alpha = 1, gamma = 0;//, beta = 1;

	cublasCreate_v2(&bhandle);

	cudaMalloc((void**)&dev_yt, M*sizeof(float));
	cudaMalloc((void**)&dev_Phit, M*M*sizeof(float));
	cudaMalloc((void**)&dev_tempA, M*M*sizeof(float));

	cudaMalloc((void***)&dev_auxArray, M*M*sizeof(float*));
	cudaMalloc((void**)&dev_pivotarrr, M*sizeof(int));
	cudaMalloc((void**)&dev_infoarr, sizeof(int));
	cudaMalloc((void***)&dev_Carray, sizeof(float*));

	cudaMemcpy(dev_yt,yt,M*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Phit, Phit, M*M*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dev_tempA, Phit, M*M*sizeof(float), cudaMemcpyDeviceToHost);

	//dev_Array[0] = dev_auxArray;

	//Convert dev_Phit to column major format first
	cublasSgemm_v2(bhandle,opT,opN,M,M,M,&alpha,dev_Phit,M,dev_Phit,M,dev_tempA,&gamma,M);

	//Invert Matrix
	cudaMemcpy(dev_auxArray, dev_tempA, M*M*sizeof(float), cudaMemcpyDeviceToDevice);

	cublasSgetrfBatched(bhandle,M,dev_ArrayLU,M,dev_pivotarrr,dev_infoarr,1); //Column Major format
	//cublasSgetriBatched(bhandle, M, dev_Array, M, dev_pivotarrr, dev_Carray, M, dev_infoarr, 1);
	cublasSgetriBatched(bhandle, M, dev_Array, M, dev_pivotarrr, dev_Carray, M, dev_infoarr, 1);

	cudaMemcpy(dev_tempA, *dev_Array, M*M*sizeof(float), cudaMemcpyDeviceToDevice);

	cublasSgemm_v2(bhandle, opN, opT, M, M, M, &alpha, dev_tempA, M, dev_Phit, M,NULL, &gamma,M);
	cublasSgemv_v2(bhandle, opN, M, M, &alpha, dev_tempA, M, dev_yt, 1, &gamma, NULL, 1);

	cublasGetVector(M, sizeof(float), Stadd, 1, dev_yt, 1);

	cublasDestroy_v2(bhandle);

	cudaFree(dev_yt);
	cudaFree(dev_Phit);
	cudaFree(dev_tempA);
	//cudaFree(dev_Array);
	cudaFree(dev_pivotarrr);
	cudaFree(dev_infoarr);
	//cudaFree(dev_Carray);

	return 0;
}
