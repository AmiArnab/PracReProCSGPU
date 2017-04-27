#include "GetSupportSet.h"

int getSupportSet(int *supportset, float *inputarr, int N)
{
	for (int i = 0; i < N; ++i)
	{
		if (inputarr[i] != 0.0) supportset[i] = 1;
		else supportset[i] = 0;
	}
	return 0;
}

float findIntersection(int *Told1, int *Told2, int M)
{
	int intersectno = 0;
	int cardinality = 0;
	for (int i = 0; i < M; ++i)
	{
		if (Told2[i] == 1) cardinality++;
		if (Told1[i] == Told2[i]) intersectno++;
	}
	return (float)(intersectno / cardinality);
}

float getSetDifference(int *Told1, int *Told2, int M)
{
	int diffcount = 0;
	int totalcount = 0;
	for (int i = 0; i < M; ++i)
	{
		if (Told1[i] == 1) totalcount++;
		if ((Told2[i] == 1) && (Told1[i] == 0)) diffcount++;
	}
	return (float)(diffcount / totalcount);
}

int getSupportCardinality(int *T, int M)
{
	int count = 0;
	for (int i = 0; i < M; ++i)
	{
		if (T[i] == 1)count++;
	}
	return count;
}
float computeOmega(float *dev_Mt, int M)
{
	cublasHandle_t chandle;
	float host_omega;
	cublasCreate_v2(&chandle);
	//float *dev_Mt;
	//cudaMalloc((void**)&dev_Mt, M*sizeof(float));
	//cudaMemcpy(dev_Mt, Mt, M*sizeof(float), cudaMemcpyDeviceToDevice);
	cublasSdot_v2(chandle, M, dev_Mt, 1, dev_Mt, 1, &host_omega);
	//cudaFree(dev_Mt);
	host_omega = host_omega / M;
	host_omega = sqrtf(host_omega);

	cublasDestroy_v2(chandle);

	return host_omega;
}
int computeLt(float *dev_Lt, float *dev_Mt, float *dev_St, int M)
{
	thrust::device_vector<float> devLt(dev_Lt,dev_Lt+M-1);
	thrust::device_vector<float> devMt(dev_Mt,dev_Mt+M-1);
	thrust::device_vector<float> devSt(dev_St,dev_St+M-1);
	for (int i = 0; i < M; i++)
	{
		devLt[i] = devMt[i] - devSt[i];
	}
	return 0;
}
