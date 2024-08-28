#pragma once
#include <cmath>
#include <utility>
#include "myUtility.h"

class Activation
{
public:
	virtual float calculate(float x) = 0;
	virtual float calculateGradient(float x) = 0;
};

class sigmoid : public Activation
{
public:
	float calculate(float x) override;
	float calculateGradient(float x) override;
};

class relu : public Activation
{
public:
	float calculate(float x) override;
	float calculateGradient(float x) override;
};

class elu : public Activation
{
public:
	elu(float alpha);
	float calculate(float x) override;
	float calculateGradient(float x) override;
private:
	float mAlpha;
};

class hyperbolictan : public Activation
{
public:
	float calculate(float x) override;
	float calculateGradient(float x) override;
};
