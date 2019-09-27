//This program is written by Abubakr Shafique (abubakr.shafique@gmail.com) 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Equalization_CUDA.h"

__global__ void Calculate_Min_Max(unsigned char* Image, int Channels, int* Min, int* Max);
__global__ void Histogram_Equalization(unsigned char* Image, int Channels, int* Min, int* Max);
__device__ int New_Pixel_Value(int Value, int Min, int Max);

void Histogram_Equalization_CUDA(unsigned char* Image, int Height, int Width, int Channels){
	unsigned char* Dev_Image = NULL;
	int* Dev_Min = NULL;
	int* Dev_Max = NULL;

	//allocate cuda variable memory
	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Min, Channels * sizeof(int));
	cudaMalloc((void**)&Dev_Max, Channels * sizeof(int));

	int Min[3] = {255, 255, 255};
	int Max[3] = {0, 0, 0};

	//copy CPU data to GPU
	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Min, Min, Channels * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Max, Max, Channels * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Calculate_Min_Max << <Grid_Image, 1 >> >(Dev_Image, Channels, Dev_Min, Dev_Max);
	Histogram_Equalization << <Grid_Image, 1 >> >(Dev_Image, Channels, Dev_Min, Dev_Max);

	//copy memory back to CPU from GPU
	cudaMemcpy(Image, Dev_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	//free up the memory of GPU
	cudaFree(Dev_Image);
}

__global__ void Calculate_Min_Max(unsigned char* Image, int Channels, int* Min, int* Max){
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * Channels;
	for (int i = 0; i < Channels; i++){
		atomicMin(&Min[i], Image[Image_Idx + i]);
		atomicMax(&Max[i], Image[Image_Idx + i]);
	}
}

__global__ void Histogram_Equalization(unsigned char* Image, int Channels, int* Min, int* Max){
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = (x + y * gridDim.x) * Channels;
	for (int i = 0; i < Channels; i++){
		Image[Image_Idx + i] = New_Pixel_Value(Image[Image_Idx + i], Min[i], Max[i]);
	}
}

__device__ int New_Pixel_Value(int Value, int Min, int Max){
	int Target_Min = 0;
	int Target_Max = 255;

	return (Target_Min + (Value - Min) * (int)((Target_Max - Target_Min)/(Max - Min)));
}
