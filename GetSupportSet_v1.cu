
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
using namespace std;

int getSupportSet(int *supportset, float *inputarr, int N)
{
	for (int i = 0; i < N; ++i)
	{
		if (inputarr[i] != 0.0) supportset[i] = 1;
		else supportset[i] = 0;
	}
	return 0;
}

int main()
{
	int N = 10;
	int *sup = new int[N];
	float *arr = new float[N];

	for (int i = 0; i < N; ++i)
	{
		arr[i] = i;
	}

	arr[2] = 0;
	arr[5] = 0;
	arr[7] = 0;

	getSupportSet(sup, arr, N);

	for (int i = 0; i < N; ++i)
	{
		cout << i << " : " << sup[i] << endl;
	}

	getchar();
	delete[] arr;
	delete[] sup;
    return 0;
}