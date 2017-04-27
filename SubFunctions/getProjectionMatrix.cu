#include "getProjectionMatrix.h"

int getProjectionMat(float *dev_Phi, float *dev_P, int M, int N)
{
	float alpha = 1;
	//float beta = 0;
	float gamma = -1;
	//float *dev_I = NULL;

	cublasHandle_t chandle;
	cublasOperation_t topA = CUBLAS_OP_N;
	cublasOperation_t topB = CUBLAS_OP_T;

	//cudaMalloc((void **)&dev_I, M*M*sizeof(float));
	thrust::device_vector<float> devI(M*M);
	thrust::fill(devI.begin(),devI.end(),0);
	for(int i=0;i<M;++i)
	{
		devI[i*M+i] = 1;
	}

	cublasCreate_v2(&chandle);
	cublasSgemm(chandle, topA, topB, M, M, N,&gamma,dev_P,M,dev_P,M,&alpha,thrust::raw_pointer_cast(&devI[0]),M);
	cudaMemcpy(dev_Phi,thrust::raw_pointer_cast(&devI[0]),M*M*sizeof(float),cudaMemcpyDeviceToDevice);

	//cublasSgemm(chandle, topA, topB, M, M, N,&alpha,dev_P,M,dev_P,M,&beta,dev_Phi,M);
	//cublasSgeam(chandle, topA, topA, M, N, &alpha, thrust::raw_pointer_cast(&devI[0]), M, &gamma, dev_Phi, M, dev_Phi, M);

	cublasDestroy_v2(chandle);

	return 0;
}
