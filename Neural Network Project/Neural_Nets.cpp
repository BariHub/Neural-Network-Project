#include "Neural_Nets.h"
#include "Functions.h"
#include <iostream>
#include <fstream>

Neural_Network::Layer::Layer(int Num_Node, Activation function):num_Nodes(Num_Node), function(function)
{
	if (Num_Node <= 0) { // check if num_node is non-zero
		std::cout << "Error in layers class constructor, number of nodes for layer is 0 or lower.\n";
		exit(0);
	}

	// Initialize outputs of each node in layer and check for NULL
	output = new float[Num_Node + 1];
	dEdO = new float[Num_Node + 1];

	for (int i = 0; i <= Num_Node; i++) {
		output[i] = 0.0;
		dEdO[i] = 0.0;
		/*if (output[i] == NULL || dEdO[i] == NULL) {
			std::cout << "Could not allocate memory for output or dEdO in the layers class constructor...\n";
			exit(0);
		}*/
	}
}

Neural_Network::Layer::~Layer()
{
	delete[] output;
	output = nullptr;
	delete[] dEdO;
	dEdO = nullptr;
	/*for (int i = 0; i <= num_Node; i++) {
		delete[] weights[i];
		weights[i] = nullptr;
		delete[] dEdW[i];
		dEdW[i] = nullptr;
	}
	delete[] weights;
	weights = nullptr;
	delete[] dEdW;
	dEdW = nullptr;*/
}

int Neural_Network::Layer::Get_NumNode() const
{
	return num_Nodes;
}

Neural_Network::Layer::Activation Neural_Network::Layer::Get_Activation() const
{
	return function;
}

//Neural_Network::Neural_Network(const Neural_Network& NN)
//{}

Neural_Network::Neural_Network(int Num_L, int Num_Node[], Layer::Activation function)
{
	num_Layers = Num_L;
	grad_Step = 1.0;

	if (Num_L <= 0) {
		std::cout << "Error in neural network class constructor, number of layers is less than 0.\n";
		exit(0);
	}
	// Create layers in an array of pointers to layer object, the for loop is fine.
	for (int i = 1; i <= Num_L; i++) {
		Layers[i] = new Layer(Num_Node[i - 1], function); // this is fine
		if (Layers[i] == NULL) {
			std::cout << "Error: Could not create layer in neural network class constructor...\n";
			exit(0);
		}
	}
	// Initialize the double pointer arrays of weights and DEDW [current][previous]
	for (int L = 2; L <= Num_L; L++) {
		Layers[L]->weights = new float* [Layers[L]->num_Nodes + 1];
		Layers[L]->dEdW = new float* [Layers[L]->num_Nodes + 1];
		for (int i = 0; i <= Layers[L]->num_Nodes; i++) { // including bias
			Layers[L]->weights[i] = new float[Layers[L - 1]->num_Nodes + 1];
			Layers[L]->dEdW[i] = new float[Layers[L - 1]->num_Nodes + 1];
		}
	}
	// Check for error and initialize values for weights and DEDW
	for (int L = 2; L <= Num_L; L++) {
		for (int i = 1; i <= Layers[L]->num_Nodes; i++) {
			for (int j = 0; j <= Layers[L - 1]->num_Nodes; j++) {
				if (Layers[L]->weights[i][j] == NULL || Layers[L]->dEdW[i][j] == NULL) {
					std::cout << "Error in neural network constructor, could not allocate memory for weights or DEDW.\n";
					exit(0);
				}
				Layers[L]->weights[i][j] = MyRandom<float>(-1.0, 1.0);
				Layers[L]->dEdW[i][j] = 0.0;
			}
		}
	}
	// Declare num of inputs/outputs and initialize them
	inputNodes = Layers[1]->num_Nodes;
	inputs = new float[inputNodes + 1];
	outputNodes = Layers[Num_L]->num_Nodes;
	outputs = new float[outputNodes + 1];
	network_Error = new float[outputNodes + 1];

	for (int i = 0; i <= inputNodes; i++) {
		inputs[i] = 0.0;
	}
	for (int i = 0; i <= outputNodes; i++) {
		outputs[i] = 0.0;
		network_Error[i] = 0.0;
	}
}

Neural_Network::Neural_Network(const char* file_name[])
{}

Neural_Network::~Neural_Network()
{
	for (int i = 1; i <= num_Layers; i++) {
		if(Layers[i] != NULL)
		delete Layers[i];
		Layers[i] = NULL;
	}
	//delete[] Layers;
	//Layers = nullptr;
	//delete[] network_Error;
	delete[] inputs;
	delete[] outputs;
	delete[] network_Error;
}

int Neural_Network::Get_NumLayers() const
{
	return num_Layers;
}

void Neural_Network::Set_Gradient(float grad)
{
	grad_Step = grad;
}

int Neural_Network::Get_NumInputs() const
{
	return inputNodes;
}

int Neural_Network::Get_NumOutputs() const
{
	return outputNodes;
}
// Calculate the output of each layer in the network, takes in inputs and takes in an empty output to store to - DONE
void Neural_Network::ForwardPropagation(float NN_in[], float dst[])
{
	// Inputs equal to outputs of the input layer
	for (int i = 1; i <= inputNodes; i++) {
		inputs[i] = NN_in[i - 1];
		Layers[1]->output[i] = inputs[i];
	}
	// Calculate all outputs of every layer
	for (int i = 1; i < num_Layers; i++) {
		node_Outputs(Layers[i + 1], Layers[i]);
	}
	// Store the outputs of output layer in array
	for (int i = 1; i <= outputNodes; i++) {
		outputs[i] = Layers[num_Layers]->output[i];
		dst[i - 1] = outputs[i];
	}
}
// Calculate the errors such as DEDO, DEDW
void Neural_Network::BackPropagation(float real_Output[])
{
	float sum;
	// Errors of output in the output layer
	for (int i = 1; i <= Layers[num_Layers]->Get_NumNode(); i++) { // DEDO = f(xi) - y(i)
		Layers[num_Layers]->dEdO[i] = Layers[num_Layers]->output[i] - real_Output[i - 1];
	}
	// Based on the error of output layer, back propagate, dE/dO(L) = SUM(dE/dO(L+1))*(dO(L+1)/dnet(L+1))*(dnet(L+1)/dO(L))
	for (int L = num_Layers - 1; L >= 2; L--) {
		switch (Layers[L]->function) {
		case(Layer::Activation::SIGMOID):
			for (int i = 1; i <= Layers[L]->Get_NumNode(); i++) {
				sum = 0.0;
				for (int k = 1; k <= Layers[L + 1]->Get_NumNode(); k++) {
					sum += Layers[L + 1]->dEdO[k] * Layers[L + 1]->output[k] * (1 - Layers[L + 1]->output[k]) * Layers[L + 1]->weights[k][i];
				}
				Layers[L]->dEdO[i] = sum;
			}
			// Once dE/dO is calculated for each node, find dE/dW(L) = (dE/dO(L))*(dO(L)/dnet)*(dnet/dW(L))
			for (int L = 2; L <= num_Layers; L++) {
				Layers[L - 1]->output[0] = 1;
				for (int i = 1; i <= Layers[L]->Get_NumNode(); i++) {
					for (int j = 0; j <= Layers[L - 1]->Get_NumNode(); j++) {
						Layers[L]->dEdW[i][j] = Layers[L]->dEdO[i] * Layers[L]->output[i] * (1 - Layers[L]->output[i]) * Layers[L - 1]->output[j];
					}
				}
			}
			break;
		case(Layer::Activation::BINARY_STEP):
			break;
		}
	}
}

void Neural_Network::Train(float inputs[], float real_Output[])
{
	float NN_Out[MAXIMUM]; // Temporary array to store outputs of network
	// Calculate outputs given a single set of inputs
	ForwardPropagation(inputs, NN_Out);
	// Sum the errors of output nodes
	// Loss(real_Output, iteration);
	// Calculate errors for each outputs in each layer
	BackPropagation(real_Output);
	// Changes the weights based on gradient/learning rate
	for (int L = 2; L <= num_Layers; L++) {
		for (int i = 1; i <= Layers[L]->Get_NumNode(); i++) {
			for (int j = 0; j <= Layers[L - 1]->Get_NumNode(); j++) {
				Layers[L]->weights[i][j] -= grad_Step * Layers[L]->dEdW[i][j];
			}
		}
	}
}

void Neural_Network::Loss(float real_Output[], int iteration)
{
	// Calculate error of each output node with actual output, sum all
	for (int i = 1; i <= Layers[num_Layers]->Get_NumNode(); i++) {
		network_Error[iteration] += ((Layers[num_Layers]->output[i] - real_Output[i]) * (Layers[num_Layers]->output[i] - real_Output[i]));
	}
	network_Error[iteration] = network_Error[iteration] / Layers[num_Layers]->Get_NumNode();
}

void Neural_Network::Save_Weights(const char file_name[]) const
{
	std::ofstream fout;
	fout.open(file_name);

	/*fout << std::scientific;
	fout.precision(5);*/

	if (!fout) {
		std::cout << "Error in saving weights to a file, could not open...\n";
		return;
	}

	fout << "# of layers:" << "," << num_Layers << std::endl;
	fout << "Nodes:" << ",";
	for (int L = 1; L <= num_Layers; L++) {
		fout << Layers[L]->Get_NumNode() << ",";
	}
	fout << std::endl;

	for ( int L = 2; L <= num_Layers; L++) {
		fout << (int)Layers[L]->function << std::endl; // check
		for (int i = 1; i <= Layers[L]->Get_NumNode(); i++) {
			for (int j = 0; j <= Layers[L - 1]->Get_NumNode(); j++) {
				fout << Layers[L]->weights[i][j] << ",";
			}
			fout << std::endl;
		}
	}
	fout.close();
}

void Neural_Network::node_Outputs(Neural_Network::Layer* Current, Neural_Network::Layer* Previous) {
	float net = 0.0;
	for (int i = 1; i <= Current->num_Nodes; i++) {
		net = Current->weights[i][0];
		for (int j = 1; j <= Previous->num_Nodes; j++) {
			net += Current->weights[i][j] * Previous->output[j];
		}
		switch (Current->function) {
		case(Layer::Activation::SIGMOID):
			Current->output[i] = Sigmoid(net);
			break;
		case(Layer::Activation::BINARY_STEP):
			break;
		}
	}
}

void Neural_Network::ShowOutputs() const
{
	for (int i = 1; i <= outputNodes; i++) {
		std::cout << outputs[i] << " ";
	}
	std::cout << std::endl;
}
