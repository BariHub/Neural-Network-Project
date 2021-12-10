#include <iostream>
#include <Vector>
#include "Neural_Nets.h"
#include <fstream>

using namespace std;

const int MAX = 100;

int main()
{
	int numL = 4;
	int nodes[4] = { 1,10,10,1 };
	Neural_Network::Layer::Activation function = Neural_Network::Layer::Activation::SIGMOID;
	Neural_Network NN(numL, nodes, function);
	float xdata[MAX][1] = { {0.0},{0.1},{0.2},{0.3},{0.4},{0.5},{0.6},{0.7},{0.8},{0.9} };
	float ydata[MAX][1] = { {0.0},{0.01},{0.04},{0.09},{0.16},{0.25},{0.36},{0.49},{0.64},{0.81} };
	float xtest[MAX][1] = { {0.05},{0.15},{0.25},{0.35},{0.45},{0.55},{0.65},{0.75},{0.85},{0.95} };
	float ytest[MAX][1] = { {0.0025},{0.0225},{0.0625},{0.1225},{0.2025},{0.3025},{0.4225},{0.5625},{0.7225},{0.9025} };
	float youts[1] = { 0 };
	float inputs[MAX];
	float outputs[MAX];
	int iterations = 10000; // epochs
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < 10; j++) {
			for (int k = 0; k < NN.Get_NumInputs(); k++) {
				inputs[k] = xdata[j][k];
				outputs[k] = ydata[j][k];
			}
			NN.Train(inputs, outputs);
		}
	}
	
	for (int j = 0; j < 10; j++) {
		for (int k = 0; k < NN.Get_NumInputs(); k++) {
			inputs[k] = xtest[j][k];
		}
		NN.ForwardPropagation(inputs, youts);
		std::cout << youts[0] << endl;
	}

	std::cout << "\npress enter to continue.\n";
	getchar();
	return 0;
}

	// set the training data set
	for (int p = 1; p <= ptraining; p++) {
		x[1] = 1.0 * (p - 1) / ptraining;
		y[1] = x[1] * x[1]; // y = x^2
		for (int k = 1; k <= NN.Get_NumInputs(); k++) {
			xdata[p][k] = x[k]; 
		}
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) {
			ydata[p][k] = y[k];
		}
	}

	NN.Set_Gradient(0.1);

	ofstream fout("E_vs_iter.csv");
	if (!fout) {
		std::cout << "File did not open.\n";
		return -1;
	}

	fout << scientific;
	fout << "iteration,E\n";

	for (int i = 1; i <= iterations; i++) {
		for (int p = 1; p <= ptraining; p++) {
			for (int k = 1; k <= NN.Get_NumInputs(); k++) {
				x[k] = xdata[p][k];
				//cout << x[k] << " ";
			}
			for (int k = 1; k <= NN.Get_NumOutputs(); k++) {
				y[k] = ydata[p][k];
				//cout << y[k] << " " << endl;
			}
			NN.Train(x, y, i);
		}

		Err_train = 0;
		for (int p = 1; p <= ptraining; p++) {
			for (int k = 1; k <= NN.Get_NumInputs(); k++)  x[k] = xdata[p][k];
			for (int k = 1; k <= NN.Get_NumOutputs(); k++) y[k] = ydata[p][k];

			NN.Layer_Outputs(x, youts);

			for (int k = 1; k <= NN.Get_NumOutputs(); k++)
				Err_train += (youts[k] - y[k]) * (youts[k] - y[k]) / NN.Get_NumOutputs();
//			E_train = E_train / N_p / NN1.N_out;

		}
		fout << i << "," << Err_train << std::endl;
	}

	NN.Layer_Outputs()

	cout << "\n     xp       y_neural_net       yp\n";
	for (int p = 1; p <= ptraining; p++) {

		for (int k = 1; k <= NN.Get_NumInputs(); k++)  x[k] = xdata[p][k]; // set x to xp
		for (int k = 1; k <= NN.Get_NumInputs(); k++) y[k] = ydata[p][k]; // set y to yp

		// calculate xs
		for (int k = 1; k <= NN.Get_NumInputs(); k++)  xs[k] = (x[k] - xmin[k]) / (xmax[k] - xmin[k]);

		// calculate neural network outputs
		NN.Layer_Outputs(xs, youts);

		// need to invert equation for ys to find yout given youts
		for (int k = 1; k <= NN.Get_NumInputs(); k++)
			yout[k] = ymin[k] + (youts[k] - 0.1) * (ymax[k] - ymin[k]) / 0.8;

		// print training data inputs
		for (int k = 1; k <= NN.Get_NumInputs(); k++) cout << x[k] << " ";

		// print neural network outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) cout << yout[k] << " ";

		// print training data outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) cout << y[k] << " ";

		cout << "\n";

	} // end for p

	// save output vs input into a file for plotting

	// quick and dirty output files, TODO: check for errors, etc.
	ofstream fout_train("out_train_vs_in.csv");
	ofstream fout_test("out_test_vs_in.csv");

	fout_train << scientific;
	fout_test << scientific;

	fout_train << "x_train,y_NN_train,y_train\n";
	fout_test << "x_test,y_NN_test,y_test\n";

	// save training points plot
	for (int p = 1; p <= ptraining; p++) {

		for (int k = 1; k <= NN.Get_NumInputs(); k++)  x[k] = xdata[p][k]; // set x to xp
		for (int k = 1; k <= NN.Get_NumInputs(); k++) y[k] = ydata[p][k]; // set y to yp

		// calculate xs
		for (int k = 1; k <= NN.Get_NumInputs(); k++)  xs[k] = (x[k] - xmin[k]) / (xmax[k] - xmin[k]);

		// calculate neural network outputs
		NN.Layer_Outputs(xs, youts);

		// need to invert equation for ys to find yout given youts
		for (int k = 1; k <= NN.Get_NumInputs(); k++)
			yout[k] = ymin[k] + (youts[k] - 0.1) * (ymax[k] - ymin[k]) / 0.8;

		// print training data inputs
		for (int k = 1; k <= NN.Get_NumInputs(); k++) fout_train << x[k] << ",";

		// print neural network outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) fout_train << yout[k] << ",";

		// print training data outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) fout_train << y[k];

		fout_train << "\n";

	} // end for p

	// save test points plot
	for (int p = 1; p <= ptesting; p++) {

		for (int k = 1; k <= NN.Get_NumInputs(); k++)  x[k] = xtest[p][k]; // set x to xp
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) y[k] = ytest[p][k]; // set y to yp

		// calculate xs
		for (int k = 1; k <= NN.Get_NumInputs(); k++)  xs[k] = (x[k] - xmin[k]) / (xmax[k] - xmin[k]);

		// calculate neural network outputs
		NN.Layer_Outputs(xs, youts);

		// need to invert equation for ys to find yout given youts
		for (int k = 1; k <= NN.Get_NumInputs(); k++)
			yout[k] = ymin[k] + (youts[k] - 0.1) * (ymax[k] - ymin[k]) / 0.8;

		// print training data inputs
		for (int k = 1; k <= NN.Get_NumInputs(); k++) fout_test << x[k] << ",";

		// print neural network outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) fout_test << yout[k] << ",";

		// print training data outputs
		for (int k = 1; k <= NN.Get_NumOutputs(); k++) fout_test << y[k];

		fout_test << "\n";

	} // end for p