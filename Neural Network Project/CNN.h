#pragma once

//class Convolution_Layer {
//public:
//	int num_Filters;
//	int kernel_Size;
//	Mat input, empty;
//	Mat* weights[1024]; // Filters
//	Mat* outputs[1024]; // Outputs of every layer
//
//	Convolution_Layer(int num_Filters, int kernel_Size); // Create the number of filters with size for one layer
//	~Convolution_Layer();
//
//	void prep();
//
//	void pooling(); // Decreases resolution and extracts important features out of it, avg, max
//
//
//private:
//
//};
//
//class conv_Layer { // Filters in one convolution layer
//public:
//	int num_Filters;
//	int kernel_Size;
//	int depth;
//	int stride;
//
//	Mat* inputArray;
//	Mat** Filters;
//	Mat* outputArray;
//
//
//	conv_Layer(Mat& src, int num_Filters, int kernel_Size = 3, int stride = 1);
//	~conv_Layer();
//
//	Mat get_Filter(int select, int depth) const;
//	void display_Filter(int select, int depth) const;
//	void display_InputArray(int depth) const;
//	void conv_Outputs();
//	void display_OutputArray(int select) const;
//
//private:
//
//};
