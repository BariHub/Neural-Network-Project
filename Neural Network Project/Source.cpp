#include <iostream>
#include <Vector>
#include "Neural_Nets.h"
#include <fstream>

using namespace std;

const int MAX = 100;
Matplotlib - cpp
HDF5 data handling
PThreads
eigen

int main() {
  Sequential model;
  model.add(new Dense(128, 784));
  model.add(new Dense(64, 128));
  model.add(new Dense(10, 64));

  Optimizer* optimizer = new SGD(0.01);
  LossFunction* loss = new MeanSquaredError();

  model.compile(optimizer, loss);
  model.train(training_data, training_labels, 10, 32);

  float* predictions = model.predict(test_data);

  // Output or process predictions
  return 0;
}