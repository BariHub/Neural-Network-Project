#pragma once

/*****************************************************************
            ACTIVATION FUNCTIONS
*****************************************************************/

// if positive y = x else 0
float relu(float x);

float relu_gradient(float x);

// Any x value higher than 0 will produce a value y >= 0.5 which will activate, if below 0, it will deactivate - DONE
float sigmoid(float x);

float sigmoid_gradient(float x);

// Output value is between -1 to 1 for any x value
float HyperbolicTan(float x); // Hyperbolic_Tan

float HyperbolicTan_gradient(float x);

// return 0 or 1 based on sign of x
bool Binary_Step(float x);

// Similar to ReLU but it leaks, ELU = exponential linear unit
float ELU(float x, float alpha);