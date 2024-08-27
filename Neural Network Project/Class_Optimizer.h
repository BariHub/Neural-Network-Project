#pragma once
class Optimizer {
public:
  virtual void update(float& weight, float gradient) = 0;
};

class StochasticGradientDescent : public Optimizer {
public:
  StochasticGradientDescent(float);
  void update(float& weight, float gradient) override;

private:
  float learning_rate;
};
