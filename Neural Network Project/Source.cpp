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
  Sequential model;
  sigmoid* s = new sigmoid();
  Dense* d1 = new Dense(2, 1, s);
  Dense* d2 = new Dense(1, 1, s);
  //Dense* d3 = new Dense(100, 1);
  //d3->set_activation(sigmoid);

  float** inputs;
  float** outputs;

  inputs = new float* [4];
  outputs = new float* [4];

  for (int i = 0; i < 4; ++i)
  {
    inputs[i] = new float[2];
    outputs[i] = new float[2];
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

  model.add(d1);
  model.add(d2);
  //model.add(d3);

  Optimizer* optimizer = new StochasticGradientDescent(1.0f);
  LossFunction* loss = new MeanSquaredError(1);

  model.compile(optimizer, loss);
  model.displayWeights();

  model.train(inputs, outputs, 4, 1000, 1);

  float* predictions = model.predict(inputs[3]);
  std::cout << predictions[0];
  return 0;
}

//Confusion Matrix : Implement a confusion matrix for classification tasks to analyze where the model is making mistakes.