#include "Activation_Functions.h"

float sigmoid::calculate(float x)
{
  return 1.0f / (1.0f + exp(-x));
}

float sigmoid::calculateGradient(float x)
{
  float fnc = calculate(x);
  return fnc * (1.0f - fnc);
}

float relu::calculate(float x)
{
  return std::max(0.0f, x);
}

float relu::calculateGradient(float x)
{
  if (SIGN(x) == 1.0f) return 1.0f;
  return 0.0f;
}

elu::elu(float alpha) : mAlpha(alpha) {}

float elu::calculate(float x)
{
  if (SIGN(x) == 1.0f) return x; // y = x
  else return mAlpha * (exp(x) - 1.0f); // alpha * ( e^x - 1 ) if negative, leaky relu
}

float elu::calculateGradient(float x)
{
  if (SIGN(x) == 1.0f) return 1.0f;
  else return mAlpha * exp(x);
}

float hyperbolictan::calculate(float x)
{
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float hyperbolictan::calculateGradient(float x)
{
  return 1 - (calculate(x) * calculate(x));
}
