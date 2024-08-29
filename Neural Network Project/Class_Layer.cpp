#include "Class_Layer.h"

Dense::Dense(int input_dim, int nodes, Activation* function) : mNeurons(nodes), mInputDimension(input_dim)
{
  mFunction = function;

  weights = new float* [mNeurons]; // number of expected inputs stored into weights[x][]
  weightGradient = new float* [mNeurons];
  mLinearTransformationSum = new float[mNeurons];

  for (int i = 0; i < mNeurons; ++i)
  {
    weights[i] = new float[input_dim]; // number of neurons in each layer
    weightGradient[i] = new float[input_dim];
    mLinearTransformationSum[i] = 0.0f;

    for (int j = 0; j < input_dim; ++j)
    {
      weights[i][j] = utilities::myRand(0.0f, 1.0f);
      weightGradient[i][j] = 0.0f;
    }
  }

  bias = new float[mNeurons];
  biasGradient = new float[mNeurons];
  mOutput = new float[mNeurons];

  for (int i = 0; i < mNeurons; ++i)
  {
    bias[i] = utilities::myRand(0.0f, 1.0f);
    biasGradient[i] = 0.0f;
  }
}

float* Dense::forward(float* input)
{
  this->mInput = input;

  for (int i = 0; i < mNeurons; ++i) 
  {
    mLinearTransformationSum[i] = applyLinearTransformation(i, mInput);
    mOutput[i] = mFunction->calculate(mLinearTransformationSum[i]);
  }
  return mOutput;
}

float* Dense::backward(float* gradient)
{
  float output_BiasTerm = 1.0;
  float* output_gradient = gradient;
  float* previousOutputGradient = new float[mInputDimension];

  for (int i = 0; i < mNeurons; ++i)
  {
    // bias term is 1.0f from notes, we calculate the weight gradients first
    biasGradient[i] = output_gradient[i] * mFunction->calculateGradient(mLinearTransformationSum[i]) * output_BiasTerm;
    for (int j = 0; j < mInputDimension; ++j)
    {
      weightGradient[i][j] = output_gradient[i] * mFunction->calculateGradient(mLinearTransformationSum[i]) * mInput[j];
    }
  }

  for (int i = 0; i < mInputDimension; ++i)
  {
    previousOutputGradient[i] = 0.0f;
    for (int j = 0; j < mNeurons; ++j)
    {
      //calculate the previous layer output gradient second
      previousOutputGradient[i] += output_gradient[j] * mFunction->calculateGradient(mLinearTransformationSum[i]) * weights[j][i];
    }
  }

  return previousOutputGradient; // Return the gradient for the previous layer
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

void Dense::displayWeights() const
{
  for (int i = 0; i < mNeurons; ++i)
  {
    std::cout << "Neuron " << i << ": " << std::endl;
    std::cout << bias[i] << " ";
    for (int j = 0; j < mInputDimension; ++j)
    {
      std::cout << weights[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

void Dense::load_weights(std::ifstream& infile)
{
  return;
}

Dense::~Dense()
{
  for (int i = 0; i < mInputDimension; ++i)
  {
    delete[] weights[i];
  }
  delete[] weights;

  for (int i = 0; i < mInputDimension; ++i)
  {
    delete[] weightGradient[i];
  }
  delete[] weightGradient;

  delete[] mInput;
  delete[] mOutput;
  delete[] bias;
  delete[] biasGradient;
}

float Dense::applyLinearTransformation(int node, float* inputs) const
{
  float output;
  output = bias[node]; // this is correct, will be sum'd soon
  for (int j = 0; j < this->mInputDimension; ++j) {
    output += mInput[j] * weights[node][j];
  }
  return output;
}