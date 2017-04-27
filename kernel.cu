
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "cublas_v2.h"
#include "thrust\sort.h"
#include "thrust\device_new.h"
#include "thrust\device_vector.h"
#include "thrust\device_delete.h"
#include "thrust\host_vector.h"
#include "thrust\iterator\counting_iterator.h"
  
#include "AdditionalLibraries\readinput.h"
#include "AdditionalLibraries\CSVparser.hpp"
#include "AdditionalLibraries\CImg.h"

#include "SubFunctions\ConcatenateVecHorizontal.h"
#include "SubFunctions\getApproxBasis.h"
#include "SubFunctions\getMatrixRank.h"
#include "SubFunctions\getProjectionMatrix.h" // Change required
#include "SubFunctions\getPruneVector.h"
#include "SubFunctions\GetSupportSet.h"
#include "SubFunctions\getThresholdVector.h"
#include "SubFunctions\l1NormMinimization.h"
#include "SubFunctions\wl1NormMinimization.h"
#include "SubFunctions\LeastSquareEstimate.h"  //Change to new version
#include "SubFunctions\RecoverAndUpdate.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
//#include <Windows.h>

using namespace std;

// Actual Image Dimension 64x80 confirmed!!
// Total number of training images 1209
// Total number of measured images 1755

int showImage(thrust::device_vector<float> *imgdata, int M, int N, int rows, int cols)
{
	char c;
	cimg_library::CImg<unsigned char> image(cols,rows, 1, 1, 255);
	
	for (int i = 0; i < M; ++i)
	{
		image(i / rows, i % rows, 0) = (unsigned char)((*imgdata)[i]);
	}

	cimg_library::CImgDisplay main_disp(image, "MyImg", 0);

	//Sleep(2000);
	cin >> c;
	return 0;
}

int main()
{
	cudaError_t err;                                                              //Error Status variable
	cublasHandle_t bhandle;                                                       //CUBLAS Handle variable
	cublasOperation_t opN = CUBLAS_OP_N;                                          //CUBLAS Operation type
	cublasOperation_t opT = CUBLAS_OP_T;                                          //CUBLAS Operation type
	float alpha = 1, beta = 1, gamma = 1;// , one = 1, zero = 0;                  //CUBLAS Operation coefficient
//----------------------------------------------------------------------------------------------------------------------------------------------------------------	
	float *traindata = NULL;                                                      //Training Image Data
	float *imagedata = NULL;                                                      //Measured Image Data
	float *timagedata = NULL;                                                     //Transposed Image Data
	int Mtrain = 5120, Ntrain = 1755;                                             //Dimension of training image data
	int Mimg = 5120, Nimg = 1209;                                                 //Dimension of measured image data
	int M = 5120, N = 1209;                                                       //General dimensions -- live variables
	int rows = 64, cols = 80;                                                     //Image dimension
	int time = 0;                                                                 //Time count -- essentially number of frames left to process
	int d = 10;                                                                   //Frame interval
	int interval = 50;                                                            //Update interval
	char c;
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
	float b = 0.95;   //65                                                            //Fraction of total energy in approximate basis
	float threshold = 10;                                                      //Threshold for computing rank
	float omega = 0.1;                                                            //Initial omega value
	int rank = 0;                                                                 //Intial rank value
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
	float *dev_yt = NULL;
	float *dev_P0 = NULL;
	float *dev_Phit = NULL;
	float *dev_Mt = NULL;
	float *dev_St = NULL;
	float *dev_Stadd = NULL;
	float *dev_Stcap = NULL;
	float *dev_Lt1 = NULL;
	float *dev_Ltemp = NULL;
	float *dev_PLt = NULL;
	float epsilon = 0, lambda = 0;
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
	float *dev_traindata = NULL;                                                  //Memory pointer for training data
	float *dev_imgdata = NULL;                                                    //Memory pointer for image data  

	int *Told1 = new int[M];                                                      //Support Set for t-1 frame 
	int *Told2 = new int[M];                                                      //Suppost Set for t-2 frame
	unsigned int *Tadd = new unsigned int[M];                                     //Support Set for addtional frame
	unsigned int *T = new unsigned int[M];                                        //Support Set for running frame

	fill(Told1, Told1 + M, 0);
	fill(Told2, Told2 + M, 0);
//------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	traindata = new float[Mtrain*Ntrain];                                         //Host memory for training data
	imagedata = new float[Mimg*Nimg];                                             //Host memory for image data
	timagedata = new float[Mimg*Nimg];                                            //Host memory for transposed image data, to access individual frames

	cudaMalloc((void **)&dev_yt, M*sizeof(float));
	cudaMalloc((void **)&dev_Phit, M*M*sizeof(float));
	cudaMalloc((void **)&dev_Mt, M*sizeof(float));
	cudaMalloc((void **)&dev_St, M*sizeof(float));
	cudaMalloc((void **)&dev_Lt1, M*sizeof(float));
	cudaMalloc((void **)&dev_Ltemp, M*sizeof(float));
	cudaMalloc((void **)&dev_Stadd, M*sizeof(float));
	cudaMalloc((void **)&dev_Stcap, M*sizeof(float));
	cudaMalloc((void **)&dev_PLt, M*d*sizeof(float));
	
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	cublasCreate_v2(&bhandle);
	cout << "Reading input data...\n";
	ReadFile("curtaintraindata.csv", traindata, Mtrain, Ntrain); //5120x1755      Read training data
	cout << "Reading training data complete!\n";
	ReadFile("curtainimagedata.csv", imagedata, Mimg, Nimg); //5120x1209          Read image data
	cout << "Reading image data complete!\n";

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------	
	cout << "Allocating device memory for data...\n";
	cudaMalloc((void **)&dev_traindata, Mtrain*Ntrain*sizeof(float));                                //Allocate memory for training image data
	cudaMalloc((void **)&dev_imgdata, Mimg*Nimg*sizeof(float));                                      //Allocate memory for actual image data
	cout << "Copying data to device memory...\n";
	cudaMemcpy(dev_traindata, traindata, Mtrain*Ntrain*sizeof(float), cudaMemcpyHostToDevice);       //Copy training data from host to device
	cudaMemcpy(dev_imgdata, imagedata, Mimg*Nimg*sizeof(float), cudaMemcpyHostToDevice);             //Copy image data from host to device
	
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
	time = 0;                                                                                        //Running time variable
	cout << "Computing initial approximate basis...\n";
	GetApproxBasis(dev_P0, &M, &N, dev_traindata, Mtrain, Ntrain, b);                                //Get the approximate basis
	
	cout << M << " " << N << endl;
	cudaDeviceSynchronize();                                                                         //Synchronize
	cout << "Computing approximate basis rank...\n";
	
	getMatRank(&rank, dev_P0, threshold, M, N);                                                      //Get matrix rank of the basis
	thrust::device_vector<float> Mt(M);
	cout << rank << endl;
	cudaDeviceSynchronize();                                                                         //Synchronize
	d = 10 * rank;                                                                                   //Set frame interval for subspace update
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
	cout << "Constructing approximate basis vector for runnig subspace...\n";
	thrust::device_vector<float> devPt(M*rank);                                                      //Construct running subspace basis
	thrust::device_vector<float> devPLt(M*d);                                                        //Construct running memory to store the previous dense vectors
	thrust::device_vector<float>::iterator it = devPt.begin();                                       //Get the iterator for the running subspace
	cout << "Initializing subspace vector array to zero values...\n";
	thrust::fill(devPLt.begin(), devPLt.end(), 0);                                                   //Initialize the running dense storage memory to zeros
	cout << "Initializing running approximate basis with computed approximate basis...\n";
	cudaMemcpy(thrust::raw_pointer_cast(&devPt[0]), dev_P0, M*rank*sizeof(float), cudaMemcpyDeviceToDevice); //Copy initial approximate basis data to running subspace basis
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
	cout << "Transposing measured image data for easy access...\n";
	cublasSgeam(bhandle, opT, opN, Nimg, Mimg, &alpha, dev_imgdata, Mimg, &beta, NULL, Mimg, dev_imgdata, Nimg);     //Transpose the image data
	cout << "Constructing running measurement vector...\n";
	//thrust::device_vector<float> Mt(Mimg);                                                                     //Construct the running image vector
	cout << "measurement vector created!\n";
	thrust::device_vector<float>::iterator itmt = Mt.begin();                                               //Get the iterator for the running image vector
	thrust::device_vector<float> devimagedata(Mimg*Nimg);
	thrust::device_vector<float>::iterator itimgt = devimagedata.begin();
	devimagedata.insert(itimgt, dev_imgdata, dev_imgdata + Mimg*Nimg);
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------
	cout << "Just for check, first frame of measured data...\n";
	
	for (int i = 0; i < Mimg; ++i)
	{
		Mt[i] = devimagedata[i*Nimg];
	}
	showImage(&Mt,Mimg,Nimg,rows,cols);
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------	
//	!!!!!!!!!!!!!!!!!!!!!!!   Need to compute the support set first !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	while (time < N)                                                                                                                       //Actual loop begins
	{
		itmt = Mt.begin();
		Mt.insert(itmt, itimgt + (time*M), itimgt + ((time + 1)*M) - 1);                                                                   //Get running frame from image data
		getProjectionMat(dev_Phit, thrust::raw_pointer_cast(&devPt[0]), M, N);                                                             //Get the projection matrix
		cudaDeviceSynchronize();
        //Check dimension N changed to M
		cublasSgemv_v2(bhandle, opN, M, M, &alpha, dev_Phit, M, thrust::raw_pointer_cast(&Mt[0]), 1, &gamma, dev_yt, 1);                   //Get the projection
		cudaDeviceSynchronize();
		memcpy((void*)Told2, (void*)Told1, M*sizeof(int));                                                                                 //Copy support set data
		//getSupportSet(Told1, dev_yt, M);                                                                                                 //Not this support set
		if (findIntersection(Told1, Told2, M) < 0.5)                                                                                       //If change is more 
		{                                                                                                                                  //Then normal l1 minmization
			cublasSgemv_v2(bhandle, opN, M, N, &alpha, dev_Phit, M, dev_Lt1, 1, &gamma, dev_Ltemp, 1);                                     //Vector product with projection matrix
			cudaDeviceSynchronize();
			cublasSnrm2_v2(bhandle, M, dev_Ltemp, 1, &epsilon); //Check what it calculates                                                 //Calculate epsilon
			cudaDeviceSynchronize();
			performBregmanIterRegularization(dev_St, dev_yt, dev_Phit, epsilon, M, N);                                                     //Perform l1 norm minimization recovery
			cudaDeviceSynchronize();
			omega = computeOmega(dev_Mt, M);                                                                                               //Computer omega -- threshold
			Thresh(T, dev_St, M, omega);                                                                                                   //Perfrom thresholding and get the support set
			cudaDeviceSynchronize();
		}
		else                                                                                                                               //Else weighted l1 minimization
		{
			lambda = getSetDifference(Told1, Told2, M);                                                                                    //Compute the set difference fraction
			cublasSgemv_v2(bhandle, opN, M, N, &alpha, dev_Phit, M, dev_Lt1, 1, &gamma, dev_Ltemp, 1);                                     //Vector product with projection matrix
			cudaDeviceSynchronize();                                                             
			cublasSnrm2_v2(bhandle, M, dev_Ltemp, 1, &epsilon);                                                                            //Calculate epsilon
			cudaDeviceSynchronize();
			performWtdBregmanIterRegularization(dev_St, dev_yt, dev_Phit, Told1, lambda, epsilon, M, N);                                   //Perform wighted l1 norm minimization recovery                     
			cudaDeviceSynchronize();
			Prune(Tadd, dev_St, M, 1.4*getSupportCardinality(Told1, M));                                                                   //Perform prunning to get support set
			cudaDeviceSynchronize();
			getLeastSquareEstimate(dev_Stadd, dev_yt, dev_Phit, Tadd, M, N);                                                               //Get least square estimate of St
			cudaDeviceSynchronize();
			omega = computeOmega(dev_Mt, M);                                                                                               //Computer omega -- threshold
			Thresh(T, dev_Stadd, M, omega);                                                                                                //Perfrom thresholding and get the support set
			cudaDeviceSynchronize();
		}
		
		getLeastSquareEstimate(dev_Stcap, dev_yt, dev_Phit, T, M, N);                                                                      //Get final least square estimate of St 
		cudaDeviceSynchronize();
		computeLt(dev_Lt1, dev_Mt, dev_Stcap, M);                                                                                          //Compute Lt
		
		bool dosvd = (time%interval)?false:true;                                                                                           //Check whether subspace needs to be updated
		UpdateSubspace(&devPt, dev_Lt1, &devPLt, M, N, d, rank, dosvd);                                                                    //Update the subspace
		cudaDeviceSynchronize();
		time++;                                                                                                                            //Increase the time count
	}

	cin >> c;

	cublasDestroy_v2(bhandle);
	//cusolverDnDestroy(my_handle);

	cudaFree(dev_yt);
	cudaFree(dev_Phit);
	cudaFree(dev_Mt);
	cudaFree(dev_Lt1);
	cudaFree(dev_Ltemp);
	cudaFree(dev_St);
	cudaFree(dev_Stadd);
	cudaFree(dev_Stcap);
	cudaFree(dev_PLt);
	cudaFree(dev_traindata);
	cudaFree(dev_imgdata);

	delete[] traindata;
	delete[] imagedata;
	delete[] timagedata;
	delete[] Told1;
	delete[] Told2;
	delete[] Tadd;
	delete[] T;

	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
