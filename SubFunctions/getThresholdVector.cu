#include "getThresholdVector.h"

int Thresh(unsigned int *T, float *dev_St, int M,float omega)
{
        thrust::device_vector<float> rsvectors(dev_St,dev_St+M);
	for (int i = 0; i < M; ++i)
	{
		if (abs(rsvectors[i]) >= omega)
		{
			T[i] = 1;
		}
		else
		{
			T[i] = 0;
		}
	}
	return 0;
}
