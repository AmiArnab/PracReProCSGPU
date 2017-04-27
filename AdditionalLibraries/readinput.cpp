#include "readinput.h"

int ReadFile(std::string myfile, float *imgdata, int M, int N)
{
	csv::Parser file(myfile.c_str());

	for (int i = 0; i < N; ++i)
	{
		imgdata[i] = 0;
	}

	for (int i = 1; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			imgdata[(i*N) + j] = std::strtof((file[i-1][j]).c_str(),0);
		}
	}
	return 0;
}