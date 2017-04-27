#include "getMatrixRank.h"

int getMatRank(int *matrank, float *dev_Mt, float threshold, int M, int N)
{
	cusolverDnHandle_t my_handle;
	cusolverDnCreate(&my_handle);

	float *dev_S = NULL;
	float *dev_U = NULL;
	float *dev_V = NULL;

	float *work = 0;
	int work_size = 0;
	int *dev_info = 0;
	int info = 0;
	
	cudaMalloc((void**)&dev_S, N * sizeof(float));
	cudaMalloc((void**)&dev_U, M * M * sizeof(float));
    	cudaMalloc((void**)&dev_V, N * N * sizeof(float));

	cusolverDnSgesvd_bufferSize(my_handle, M, N, &work_size);
	
	cudaMalloc((void**)&work, work_size * sizeof(float));
	cudaMalloc((void**)&dev_info, sizeof(int));
	cout << "Before SVD\n";
	cusolverStatus_t stat = cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_Mt, M, dev_S, dev_U, M, dev_V, N, work, work_size, NULL, dev_info);
	//cusolverStatus_t stat = cusolverDnSgesvd(my_handle, 'A', 'A', M, N, dev_Mt, M, dev_S, dev_U, M, NULL, N, work, work_size, NULL, dev_info);
	switch(stat)
	{
		case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
		case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
		case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
		case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
	}

	cudaDeviceSynchronize();
	cout << "After SVD\n";
        //thrust::device_vector<float> sigvals(dev_S,dev_S+N);
	//thrust::host_vector<float> tempval(N);
	//cudaMemcpy(thrust::raw_pointer_cast(&tempval[0]),dev_S,N*sizeof(float),cudaMemcpyDeviceToHost);
	//thrust::device_vector<float> sigs = tempval;
	//cudaMemcpy(&info,dev_info,sizeof(int),cudaMemcpyDeviceToHost);
	//cout << info << endl;
	/*cudaMemcpy(thrust::raw_pointer_cast(&sigvals[0]),dev_S,N*sizeof(float),cudaMemcpyDeviceToDevice);
        cout << "After Vector\n";
	for(int i=0;i<N;++i)
	{
	        if(sigvals[i] > threshold)
	        {
	                (*matrank)++;
	        }
	}*/
        cout << "After rank\n";
	cusolverDnDestroy(my_handle);
	cudaFree(dev_V);
	cudaFree(dev_S);
	cudaFree(dev_U);
	cudaFree(work);
	cudaFree(dev_info);
	
        return 0;
}
