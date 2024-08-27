#include "Class_Model.h"

void Sequential::add(Layer* layer)
{
  layers.push_back(layer);
}

void Sequential::compile(Optimizer* optimizer, LossFunction* loss_function)
{
  this->optimizer = optimizer;
  this->loss_function = loss_function;
}

void Sequential::train(float** inputs, float** targets, int epochs, int batch_size = 1)
{
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Implement batch training loop
    for (int batch = 0; batch < batch_size; ++batch) { // potentially wrong?
      // Forward and backward pass
      float* predictions = forward(inputs[batch]);
      backward(targets[batch], predictions);
      update_weights();
    }
  }
}

float* Sequential::predict(float* input)
{
  return forward(input);
}

float* Sequential::forward(float* input)
{
  float* output = input;
  for (Layer* layer : layers) {
    output = layer->forward(output);
  }
  return output;
}

void Sequential::backward(float* target, float* prediction)
{
  float* gradient = loss_function->compute_loss_gradient(target, prediction);

  for (std::vector<Layer*>::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it) 
  {
    gradient = (*it)->backward(gradient);
  }
}

void Sequential::update_weights()
{
  for (Layer* layer : layers) {
    layer->update_weights(optimizer);
  }
}