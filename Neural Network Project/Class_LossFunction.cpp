#include "Class_LossFunction.h"

MeanSquaredError::MeanSquaredError(int iOutputSize) : mOutputSize(iOutputSize) {}

float MeanSquaredError::compute_loss(float* target, float* prediction)
{
  float loss = 0.0f;
  for (int i = 0; i < mOutputSize; ++i) {
    float diff = target[i] - prediction[i];
    loss += diff * diff;
  }
  return loss / mOutputSize;
}

float* MeanSquaredError::compute_loss_gradient(float* target, float* prediction)
{
  float* gradient = new float[mOutputSize];
  for (int i = 0; i < mOutputSize; ++i) {
    gradient[i] = 2.0f * (prediction[i] - target[i]) / mOutputSize;
  }
  return gradient;
}
