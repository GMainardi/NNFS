import numpy as np

class Activation:
  
  def forward(self, inputs):
    pass
  
  def backward(self, dvalues):
    pass
  
class ReLU(Activation):

  def forward(self, inputs):

    self.inputs = inputs
    self.output = np.maximum(0, inputs)

    return self.output


  def backward(self, dvalues):

    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

    return self.dinputs

class Softmax(Activation):
  
    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis = 1,
                                            keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

        return self.output

    # not used, just for learning propouses
    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                    enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                                np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self.dinputs