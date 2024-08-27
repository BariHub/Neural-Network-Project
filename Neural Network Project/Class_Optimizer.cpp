#include "Class_Optimizer.h"

StochasticGradientDescent::StochasticGradientDescent(float learning_rate) : learning_rate(learning_rate) {}

void StochasticGradientDescent::update(float& weight, float gradient)
{
  weight -= learning_rate * gradient;
}
