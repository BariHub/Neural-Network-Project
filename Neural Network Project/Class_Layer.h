#pragma once
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include "Activation_Functions.h"
#include "Class_Optimizer.h"
#include "myUtility.h"

class Layer
{
public:
  virtual float* forward(float* input) = 0;
  virtual float* backward(float* gradient) = 0;
  virtual void update_weights(Optimizer*) = 0;
};

class Dense : public Layer
{
public:
  Dense(int input_dim, int nodes, Activation* function);
  float* forward(float* input) override;
  float* backward(float* gradient) override;
  void update_weights(Optimizer* optimizer) override;
  void displayWeights() const;
  void load_weights(std::ifstream& infile);
  //void save_weights(std::ifstream& infile);

  ~Dense();
private:
  int mNeurons;
  int mInputDimension;
  float* mInput;
  float* mOutput;
  float** weights;
  float** weightGradient;
  float* bias;
  float* biasGradient;
  float* mLinearTransformationSum;
  Activation* mFunction;
  float applyLinearTransformation(int node, float* inputs) const;
};
