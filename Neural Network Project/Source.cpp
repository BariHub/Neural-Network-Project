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
  return 0;
}

Confusion Matrix : Implement a confusion matrix for classification tasks to analyze where the model is making mistakes.