#include "Activation_Functions.h"
#include "myUtility.h"
#include <cmath>
#include <utility>

float relu(float x)
{
  return std::max(0.0f, x);
}

float relu_gradient(float x)
{
  if (SIGN(x) == 1.0f) return 1.0f;
  return 0.0f;
}

float sigmoid(float x)
{
  return 1.0f / (1.0f + exp(-x));
}

float sigmoid_gradient(float x)
{
  float fnc = sigmoid(x);
  return fnc * (1.0f - fnc);
}

float HyperbolicTan(float x) // Hyperbolic_Tan
{
  return 2.0f / (1.0f + exp(-2.0f * x)) - 1.0f; // f(x) = tanh(x) = (2 / ( 1 + e^(-2x) )) - 1
}

float HyperbolicTan_gradient(float x)
{
  return NULL;
}

bool Binary_Step(float x)
{
  if (SIGN(x) == 1.0f) return true; //return 1 if positive
  else return false; // return 0 if negative
}

float ELU(float x, float alpha)
{
  if (SIGN(x) == 1.0f) return x; // y = x
  else return alpha * (exp(x) - 1.0f); // alpha * ( e^x - 1 ) if negative, leaky relu
}