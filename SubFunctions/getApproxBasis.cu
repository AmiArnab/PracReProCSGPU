#include "getApproxBasis.h"

int GetApproxBasis(float *dev_approx_basis, int *Ma, int *Na, float *dev_Mtrain, int M, int N, float b)
{
	cusolverDnHandle_t my_handle;
	cusolverDnCreate(&my_handle);

	//int Malocal = 0;
	//int Nalocal = 0;
	int maxindex = 0;
	float sum = 0, maxsum = 0;

	float *dev_S = NULL;
	float *dev_U = NULL;
	float *dev_V = NULL;

	float *work = 0;
	int work_size = 0;
	int *dev_info = 0;

	cudaMalloc((void**)&dev_S, N * sizeof(float));
	cudaMalloc((void**)&dev_U, M * M * sizeof(float));
	cudaMalloc((void**)&dev_V, N * N * sizeof(float));

	//cusolverStatus_t cusolverstatus; // Not used

	cusolverDnSgesvd_bufferSize(my_handle, M, N, &work_size);
	cudaMalloc((void**)&work, work_size * sizeof(float));
	cudaMalloc((void**)&dev_info, sizeof(int));

	cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_Mtrain, M, dev_S, dev_U, M, dev_V, N, work, work_size, NULL, dev_info);
	cudaDeviceSynchronize();
	//cusolverDnDestroy(my_handle);
	thrust::device_vector<float> svalues(dev_S,dev_S+N);
	for(int i=0;i<N;++i) //changed to N
	{
	        maxsum+=svalues[i];
	}

        for(int i=0;i<N;++i) //changed to N
	{
	        sum+=svalues[i];
	        if((sum/maxsum)>b)
	        {
	                maxindex = i;
	                break;
	        }
	}

	cudaMalloc((void**)&dev_approx_basis, M * maxindex * sizeof(float));
	cudaMemset(dev_approx_basis,0,M*maxindex*sizeof(float));
	thrust::device_vector<float> rsvectors(dev_U,dev_U+(M*M));
	//thrust::device_vector<float> apbvectors(dev_approx_basis,dev_approx_basis+(M*maxindex));
	thrust::device_vector<float> apbvectors(M*maxindex);
	thrust::copy(rsvectors.begin(), rsvectors.begin()+(M*maxindex), apbvectors.begin());
	cudaMemcpy(dev_approx_basis, thrust::raw_pointer_cast(&apbvectors[0]), M*maxindex*sizeof(float), cudaMemcpyDeviceToDevice);
	*Ma = M;
	*Na = maxindex;

	cusolverDnDestroy(my_handle);
        
	cudaFree(dev_S);
	cudaFree(dev_U);
	cudaFree(dev_V);
	
	return 0;
}
