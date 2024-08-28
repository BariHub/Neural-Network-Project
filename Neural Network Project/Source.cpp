#include <iostream>
#include "Class_Model.h"
#include "Activation_Functions.h"

using namespace std;
//Matplotlib - cpp
//HDF5 for storage https://github.com/HDFGroup/hdf5
//PThreads
//eigen


int main() 
{
  float** inputs;
  float** outputs;
  inputs = new float* [4];
  outputs = new float* [4];

  for (int i = 0; i < 4; ++i)
  {
    inputs[i] = new float[2];
    outputs[i] = new float[1];
  }

  inputs[0][0] = 0.0f;
  inputs[0][1] = 0.0f;
  outputs[0][0] = 0.0f;

  inputs[1][0] = 0.0f;
  inputs[1][1] = 1.0f;
  outputs[1][0] = 0.0f;

  inputs[2][0] = 1.0f;
  inputs[2][1] = 0.0f;
  outputs[2][0] = 0.0f;

  inputs[3][0] = 1.0f;
  inputs[3][1] = 1.0f;
  outputs[3][0] = 1.0f;

  Sequential model;
  Dense* d1 = new Dense(2, 10);
  Dense* d2 = new Dense(10, 1);
  //Dense* d3 = new Dense(100, 1);

  d1->set_activation(sigmoid);
  d2->set_activation(sigmoid);
  //d3->set_activation(sigmoid);

  model.add(d1);
  model.add(d2);
  //model.add(d3);

  Optimizer* optimizer = new StochasticGradientDescent(1.0f);
  LossFunction* loss = new MeanSquaredError(1);

  model.compile(optimizer, loss);
  model.displayWeights();

  model.train(inputs, outputs, 4, 1000, 1);

  float* predictions = model.predict(inputs[3]);


  std::cout << "Input: " << inputs[3][0] << " " << inputs[3][1] << " - Target: " << outputs[3][0] << " - Prediction : " << predictions[0] << std::endl;
  // Output or process predictions
  return 0;
}