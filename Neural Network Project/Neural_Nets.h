#pragma once

const int MAXIMUM = 100;

class Neural_Network {
public:
	class Layer {
	public:
		friend class Neural_Network;
		enum class Activation { SIGMOID, BINARY_STEP, HYPERBOLIC_TAN, RELU };

		Layer(const Layer&) = delete;
		Layer& operator=(const Layer&) = delete;
		Layer(int Num_Node, Activation function); // constructor
		~Layer(); // destructor
		int Get_NumNode() const; // gets the number of nodes in layer
		Activation Get_Activation() const;
		//void set_numNode(int node); // sets the number of nodes in layer

	private:
		int num_Nodes;
		Activation function;
		float* output; // output for each node for layer
		float* dEdO;
		float** weights;
		float** dEdW;
	};

public:
	//Neural_Network(const Neural_Network& NN); // not yet implemented
	Neural_Network& operator=(const Neural_Network&) = delete;
	Neural_Network(int Num_L, int Num_Node[], Layer::Activation function); // constructor
	Neural_Network(const char* file_name[]); // constructor
	~Neural_Network();

public:
	int Get_NumLayers() const;
	//int Get_Input
	void Set_Gradient(float grad); // set the gradient
	int Get_NumInputs() const;
	int Get_NumOutputs() const;
	void ForwardPropagation(float NN_in[], float dst[]); // should be the memory location &?, NN_out is useless
	void BackPropagation(float real_Output[]); // calculate errors based on the output layer
	void Train(float inputs[], float real_Output[]); // train the network for one data point
	void Loss(float real_Output[], int iteration); // sum of error for each output node for one data point
	void Save_Weights(const char file_name[]) const; // save weights to a .csv file
	//void Export_Error(const char file_name[]) const; // try for .csv file
	void node_Outputs(Neural_Network::Layer* Current, Neural_Network::Layer* Previous);
	void ShowOutputs() const;

private:
	int num_Layers; // number of layers in the network, including the input and output layer
	float* network_Error; // error of the network for one data point

	int inputNodes; // number of input nodes in the input layer
	float* inputs; // inputs to the input layer of the network

	int outputNodes; // number of output nodes in the output layer
	float* outputs; // output layer of the network

private:
	Layer* Layers[MAXIMUM]; // pointer to layer object
	float grad_Step; // learning rate, can make it dynamic
};