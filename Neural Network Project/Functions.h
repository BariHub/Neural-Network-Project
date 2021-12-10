#pragma once
#include <cstdlib>
#include "Neural_Nets.h"

#define SIGN(a) ((a) >= 0.0 ? 1.0 : -1.0)
/*****************************************************************
						OTHER FUNCTIONS
*****************************************************************/

// Calculate rand values within a specific range - DONE
template<typename T>
double MyRandom(T min, T max)
{
	if (max > min) return ((T)rand() / (T)RAND_MAX) * (max - min) + min;
	else return ((T)rand() / (T)RAND_MAX) * (min - max) + max;
}

// Print out the output values - test
//void print_Outputs(Neural_Network& NN) {
//	for (int i = 1; i <= NN.outputNodes; i++) {
//		std::cout << "Output Node " << i << ": " << NN.output_Layer[i] << std::endl;
//	}
//}

// Bilinear interpolation - https://en.wikipedia.org/wiki/Bilinear_interpolation
/*Mat bilinear_Filter(Mat& src, Mat& dest) {
	//f(x,y1) = (x2-x)/(x2-x1)*f(Q11) + (x-x1)/(x2-x1)*f(Q21)
	//f(x,y1) = (x2-x)/(x2-x1)*f(Q12) + (x-x1)/(x2-x1)*f(Q22)
	//f(x,y) = (y2-y)/(y2-y1)*f(x,y1) + (y-y1)/(y2-y1)*f(x,y2)
	double constant = 1 / ((dest.cols) * (dest.rows));
	double x[2];
	double f[2][2] = { {src.at<float>(1,0), src.at<float>(0,0)}, {src.at<float>(1,1), src.at<float>(0,1)} };
	double y[2];

	for (int i = 0; i < dest.rows; i++) {
		for (int j = 0; j < dest.cols; j++) {

		}
	}
	return src;
}*/

/*****************************************************************
						ACTIVATION FUNCTIONS
*****************************************************************/

// Any x value higher than 0 will produce a value y >= 0.5 which will activate, if below 0, it will deactivate - DONE
float Sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

// 0 or 1 based on sign of x
bool Binary_Step(float x) {
	if (SIGN(x) == 1.0) return true; //return 1 if positive
	else return false; // return 0 if negative
}

// Output value is between -1 to 1 for any x value
float Hyperbolic_Tan(float x) {
	return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0; // f(x) = tanh(x) = (2 / ( 1 + e^(-2x) )) - 1
}

// if positive y = x else 0
float ReLU(float x) {
	if (SIGN(x) == 1.0) return x; // y = x
	else return 0.0;
}

// Similar to ReLU but it leaks, ELU = exponential linear unit
float ELU(float x, float alpha) {
	if (SIGN(x) == 1.0) return x; // y = x
	else return alpha * (exp(x) - 1.0); // alpha * ( e^x - 1 ) if negative, leaky relu
}

// Import weights from a .csv file, 
/*void import_Weights(const char file_name[]) {
	int num_l;
	string temp_line;
	ifstream fin;
	fin.open(file_name);

	fin >> scientific;
	fin.precision(5);

	if (!fin) {
		cout << "\nError in importing weights, file could not be opened...";
		return;
	}

	getline(fin, temp_line, ',');
	getline(fin, temp_line);
	num_l = stoi(temp_line);

	for (int L = 1; L <= num_l; L++) {

	}

	for (int L = 2; L <= num_Layer; L++) {
		for (int i = 1; i <= Layer[L]->get_numNode(); i++) {
			for (int j = 0; j <= Layer[L - 1]->get_numNode(); j++) {
				//fin >> Layer[L]->Weights[i][j] >> ",";
			}
			//fin >> endl;
		}
	}
	fin.close();
}*/

void normalize(float newmin, float newmax, float min, double max) {
	//Inorm = (I - min) * ((newmax - newmin) / (max - min)) + newmin;
}

/*****************************************************************
						MISCELLANEOUS FUNCTIONS
*****************************************************************/

/*std::string type2str(int type)
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}*/