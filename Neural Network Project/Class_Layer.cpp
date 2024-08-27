#include "Class_Layer.h"

Dense::Dense(int input_dim, int nodes)
{
  this->mNeurons = nodes; // number of neurons in layer x
  this->mInputDimension = input_dim; // number of inputs coming into the layer

  weights = new float* [mNeurons]; // number of expected inputs stored into weights[x][]
  weightGradient = new float* [mNeurons];

  for (int i = 0; i < mNeurons; ++i)
  {
    weights[i] = new float[input_dim]; // number of neurons in each layer
    weightGradient[i] = new float[input_dim];

    for (int j = 0; j < input_dim; ++j)
    {
      weights[i][j] = utilities::myRand(0.0f, 1.0f);
      weightGradient[i][j] = 0.0f;
    }
  }

  bias = new float[mNeurons];
  biasGradient = new float[mNeurons];
  output = new float[mNeurons];

  for (int i = 0; i < mNeurons; ++i)
  {
    bias[i] = 0.0f;
    biasGradient[i] = 0.0f;
  }
}

float* Dense::forward(float* input)
{
  this->input = input;

  for (int i = 0; i < mNeurons; ++i) {
    output[i] = bias[i]; // this is correct, will be sum'd soon
    for (int j = 0; j < this->mInputDimension; ++j) {
      output[i] += this->input[j] * weights[i][j];
    }
    // Apply activation function if set
    if (mActivationFunction) {
      output[i] = mActivationFunction(output[i]);
    }
  }
  return output;
}

float* Dense::backward(float* gradient)
{
  float output_BiasTerm = 1.0;
  float* output_gradient = gradient;
  float* previousOutputGradient = new float[mInputDimension];
  
  for (int i = 0; i < mInputDimension; ++i)
  {
    previousOutputGradient[i] = 0.0f;
  }

  for (int i = 0; i < mNeurons; ++i)
  {
    // bias term is 1.0f from notes, we calculate the weight gradients first
    biasGradient[i] = output_gradient[i] * output[i] * (1 - output[i]) * output_BiasTerm;
    for (int j = 0; j < mInputDimension; ++j)
    {
      weightGradient[i][j] = output_gradient[i] * output[i] * (1 - output[i]) * input[j];
    }
  }

  for (int i = 0; i < mInputDimension; ++i)
  {
    for (int j = 0; j < mNeurons; ++j)
    {
      //calculate the previous layer output gradient second
      previousOutputGradient[i] += output_gradient[j] * output[j] * (1 - output[j]) * weights[j][i];
    }
  }

  return previousOutputGradient; // Return the gradient for the previous layer
}

void Dense::set_activation(std::function<float(float)> activation)
{
  mActivationFunction = activation;
}

void Dense::update_weights(Optimizer* optimizer)
{
  for (int i = 0; i < mNeurons; ++i)
  {
    for (int j = 0; j < mInputDimension; ++j)
    {
      optimizer->update(weights[i][j], weightGradient[i][j]);
    }
    optimizer->update(bias[i], biasGradient[i]);
  }
}
