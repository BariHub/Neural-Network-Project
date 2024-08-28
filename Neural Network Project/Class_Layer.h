#pragma once
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include "Activation_Functions.h"
#include "Class_Optimizer.h"
#include "myUtility.h"

class Layer {
public:
  virtual float* forward(float* input) = 0;
  virtual float* backward(float* gradient) = 0;
  virtual void update_weights(Optimizer*) = 0;
};

class Dense : public Layer {
public:
  Dense(int, int);
  float* forward(float*) override;
  float* backward(float*) override;
  void set_activation(std::function<float(float)> activation);
  void update_weights(Optimizer*) override;
  void displayWeights() const;
  void load_weights(std::ifstream& infile);
  //void save_weights(std::ifstream& infile);

  ~Dense();
private:
  int mNeurons;
  int mInputDimension;
  float* input;
  float* output;
  float** weights;
  float** weightGradient;
  float* bias;
  float* biasGradient;
  std::function<float(float)> mActivationFunction;
};
