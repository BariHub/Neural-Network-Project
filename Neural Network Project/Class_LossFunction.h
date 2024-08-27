#pragma once

class LossFunction {
public:
  virtual float compute_loss(float* target, float* prediction) = 0;
  virtual float* compute_loss_gradient(float* target, float* prediction) = 0;
};

class MeanSquaredError : public LossFunction {
public:
  MeanSquaredError(int);
  float compute_loss(float* target, float* prediction) override;
  float* compute_loss_gradient(float* target, float* prediction) override;

private:
  int mOutputSize;
};
