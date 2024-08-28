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

void Sequential::train(float** inputs, float** targets, int num_DataPoints, int epochs, int batch_size = 1)
{
  for (int epoch = 0; epoch < epochs; ++epoch) 
  {
    for (int DataPoint = 0; DataPoint < num_DataPoints; ++DataPoint)
    {
      float* predictions = forward(inputs[DataPoint]);
      backward(targets[DataPoint], predictions);
      update_weights();
      displayWeights();
    }
  }
}

float* Sequential::predict(float* input)
{
  return forward(input);
}

void Sequential::displayWeights() const
{
  int i = 0;
  for (std::vector<Layer*>::const_iterator it = layers.begin(); it < layers.end(); ++it)
  {
    i++;
    std::cout << "Layer " << i << ": " << std::endl;
    dynamic_cast<Dense*>(*it)->displayWeights();
    std::cout << std::endl;
  }
}

void Sequential::load_weights(const std::string& file_path)
{
  std::ifstream infile;
  infile.open(file_path);

  if (!infile) {
    std::cerr << "Failed to open weights file: " << file_path << std::endl;
    return;
  }

  for (std::vector<Layer*>::iterator::value_type& layer : layers) {
    Dense* dense_layer = dynamic_cast<Dense*>(layer);
    if (dense_layer) {
      dense_layer->load_weights(infile);
    }
  }

  infile.close();
}

Sequential::~Sequential()
{
  delete optimizer;
  delete loss_function;
}

float* Sequential::forward(float* input)
{
  float* output = input;
  for (Layer* layer : layers) 
  {
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