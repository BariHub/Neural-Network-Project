#pragma once
#include "Class_Layer.h"
#include "Class_Optimizer.h"
#include "Class_LossFunction.h"
#include "Activation_Functions.h"
#include <vector>
#include <iostream>
#include <fstream>

class Model {
public:
  virtual void add(Layer*) = 0;
  virtual void compile(Optimizer*, LossFunction*) = 0;
  virtual void train(float**, float**, int, int) = 0;
  virtual float* predict(float*) = 0;
};

class Sequential {
public:
  void add(Layer* layer);
  void compile(Optimizer* optimizer, LossFunction* loss_function);
  void train(float** inputs, float** targets, int num_DataPoints, int epochs, int batch_size);
  float* predict(float* input);
  void displayWeights() const;
  void load_weights(const std::string& file_path);
  //void save_weights(const std::string& file_path);

  ~Sequential();
  std::vector<Layer*> layers;
private:
  Optimizer* optimizer;
  LossFunction* loss_function;
  
  float* forward(float* input);
  void backward(float* target, float* prediction);
  void update_weights();
};