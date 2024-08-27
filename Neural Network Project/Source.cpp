#include <iostream>
#include "Class_Model.h"

//Matplotlib - cpp
//HDF5 data handling
//PThreads
//eigen

int main() {

  float** inputs;
  float** outputs;
  inputs = new float* [100];
  outputs = new float* [100];

  float sum = 0.0f;

  for (int i = 0; i < 100; ++i)
  {
    inputs[i] = new float[1];
    outputs[i] = new float[1];
  }

  for (int i = 0; i < 100; ++i)
  {
    sum += 1.0f;
    inputs[i][0] = sum / 100.0f;
    outputs[i][0] = inputs[i][0] * inputs[i][0];
  }

  Sequential model;
  model.add(new Dense(1, 784));
  model.add(new Dense(784, 128));
  model.add(new Dense(128, 1));

  Optimizer* optimizer = new StochasticGradientDescent(0.001f);
  LossFunction* loss = new MeanSquaredError(1);

  model.compile(optimizer, loss);
  model.train(inputs, outputs, 10, 1);

  float* test = new float[1];
  test[0] = 0.5f;
  float target = 0.5f * 0.5f;

  float* predictions = model.predict(test);

  std::cout << "Input: " << test[0] << " - Target: " << target << " - Prediction: " << *predictions << std::endl;
  // Output or process predictions
  return 0;
}