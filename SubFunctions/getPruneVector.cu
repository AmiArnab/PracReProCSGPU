#include "getPruneVector.h"

int Prune(unsigned int *T, float *St, int M, unsigned int s)
{
	thrust::device_vector<float> pkeys(St,St+M-1);
	thrust::host_vector<int> skeys(M);
	thrust::counting_iterator<int> it(0);
	thrust::device_vector<int> indices(M);
	thrust::copy(it, it + indices.size(), indices.begin());
	thrust::sort_by_key(pkeys.begin(), pkeys.end(), indices.begin());
	thrust::copy(indices.begin(), indices.end(), skeys.begin());
	for (int i = 0; i < M; i++)
	{
		T[i] = 0;
	}
	for (int i = M; i>M - s; i--)
	{
		T[skeys[i]] = 1;
	}

	return 0;
}
