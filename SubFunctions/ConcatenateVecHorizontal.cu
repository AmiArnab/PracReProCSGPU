#include "ConcatenateVecHorizontal.h"

int ConcatenateMatrix(float **dev_outmat, float *dev_inmat, float **dev_listmat, int M, int N, int n)
{
	float *dev_newout, *dout;
	cudaMalloc((&dev_newout), (N + n)*M*sizeof(float));
	cudaMemset(dev_newout, 0, (N + n)*M*sizeof(float));
	thrust::device_vector<float> dinvec(dev_inmat, dev_inmat + (M*N));
	thrust::device_vector<float> doutvec(dev_newout, dev_newout + (N + n)*M);
	for (int j = 0; j < N; ++j)
	{
		for (int i = 0; i < M; ++i)
		{
			doutvec[(i*(N + n)) + j] = dinvec[(i*N) + j];
		}
	}

	for (int k = 0; k < n; k++)
	{
		thrust::device_vector<float> dlsvec(*(dev_listmat+k), *(dev_listmat +k) + M);
		for (int j = N; j < (N + n); ++j)
		{
			for (int i = 0; i < M; ++i)
			{
				doutvec[(i*(N + n)) + j] = dlsvec[i];
			}
		}
	}

	dout = thrust::raw_pointer_cast(doutvec.data());
	cudaMemcpy(*dev_outmat, dout, M*(N + n)*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cudaFree(dev_newout);
	return 0;
}
