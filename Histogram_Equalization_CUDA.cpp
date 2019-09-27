//This Program is written by Abubakr Shafique (abubakr.shafique@gmail.com)
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Equalization_CUDA.h"

using namespace std;
using namespace cv;

int main(){
	Mat Input_Image = imread("Low_Contrast.jpg", 0); // Read Gray Image

	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;

	Histogram_Equalization_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels());

	imwrite("Histogram_Image.png", Input_Image);
	system("pause");//to stop when every thing executes
	return 0;
}