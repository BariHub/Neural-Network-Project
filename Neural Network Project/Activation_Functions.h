#include <utility>
#include <cmath>
#include "myUtility.h"

/*****************************************************************
            ACTIVATION FUNCTIONS
*****************************************************************/

// if positive y = x else 0
float relu(float x)
{
  return std::max(0.0f, x);
}

float relu_gradient(float x)
{
  if (SIGN(x) == 1.0f) return 1.0f;
  return 0.0f;
}

// Any x value higher than 0 will produce a value y >= 0.5 which will activate, if below 0, it will deactivate - DONE
float sigmoid(float x)
{
  return 1.0f / (1.0f + exp(-x));
}

float sigmoid_gradient(float x)
{
  float fnc = sigmoid(x);
  return fnc * (1.0f - fnc);
}

// Output value is between -1 to 1 for any x value
float tanh(float x) // Hyperbolic_Tan
{
  return 2.0f / (1.0f + exp(-2.0f * x)) - 1.0f; // f(x) = tanh(x) = (2 / ( 1 + e^(-2x) )) - 1
}

float tanh_gradient(float x)
{
  return NULL;
}

// return 0 or 1 based on sign of x
bool Binary_Step(float x)
{
  if (SIGN(x) == 1.0f) return true; //return 1 if positive
  else return false; // return 0 if negative
}

// Similar to ReLU but it leaks, ELU = exponential linear unit
float ELU(float x, float alpha)
{
  if (SIGN(x) == 1.0f) return x; // y = x
  else return alpha * (exp(x) - 1.0f); // alpha * ( e^x - 1 ) if negative, leaky relu
}