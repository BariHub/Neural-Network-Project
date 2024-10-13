#include <iostream>
#include "Class_Model.h"
#include "Activation_Functions.h"
#include <fstream>

using namespace std;
//Matplotlib - cpp
//HDF5 for storage https://github.com/HDFGroup/hdf5
//PThreads
//eigen


int main() 
{
  Sequential model;
  sigmoid* s = new sigmoid();
  Dense* d1 = new Dense(1, 1, s);
  Dense* d2 = new Dense(1, 1, s);
  //Dense* d3 = new Dense(100, 1);
  //d3->set_activation(sigmoid);

  float** inputs;
  float** outputs;
  int size = 10;

  inputs = new float* [size];
  outputs = new float* [size];
  float** p;
  p = new float* [size];

  for (int i = 0; i < size; ++i)
  {
    inputs[i] = new float[1];
    outputs[i] = new float[1];
    p[i] = new float[1];
  }

  for (int i = 0; i < size; ++i)
  {
    float x = (float)(i + 1) * 0.1f;
    inputs[i][0] = x;
    outputs[i][0] = x * x;
  }
  

  for (int i = 0; i < size; ++i)
  {
    float x = (float)(i + 0.5) * 0.1f;
    p[i][0] = x;
    std::cout << p[i][0] << " ";
  }

  model.add(d1);
  model.add(d2);
  //model.add(d3);

  Optimizer* optimizer = new StochasticGradientDescent(0.01f);
  LossFunction* loss = new MeanSquaredError(1);

  model.compile(optimizer, loss);

  model.train(inputs, outputs, size, 10000, 1);

  float* pred = new float[size];
  for (int i = 0; i < size; ++i)
  {
    pred[i] = model.predict(p[i])[0];
  }

  std::ofstream file("output.csv");

  // Check if the file was successfully opened
  if (!file.is_open()) {
    std::cerr << "Error: Unable to create file" << std::endl;
    return 1;
  }

  for (int i = 0; i < size; ++i)
  {
    file << inputs[i][0] << "," << outputs[i][0] << "," << pred[i];
    file << std::endl;
  }

  file.close();

  return 0;
}

//Confusion Matrix : Implement a confusion matrix for classification tasks to analyze where the model is making mistakes.