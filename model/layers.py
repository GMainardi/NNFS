import numpy as np

class Layer:
  
    def __init__(self):
      pass
  
    def forward(self, inputs):
      pass
  
    def backward(self, dvalues):
      pass
  
class Dense(Layer):

  def __init__(self, inputs, neurons):

    self.weights = 0.01 * np.random.randn(inputs, neurons)
    self.biases = np.zeros((1, neurons))

  def forward(self, inputs):

    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

    return self.output


  def backward(self, dvalues):

    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

    self.dinputs = np.dot(dvalues, self.weights.T)

    return self.dinputs