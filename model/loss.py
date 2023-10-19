import numpy as np
from model.activations import Softmax

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss
  
class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # OHE y_true
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # categorical y_true
        elif len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        negative_log = -np.log(correct_confidences)
        return negative_log

    # not used, just for learning propouses
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        labels = len(dvalues[0])

        # to OHE
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        self.dinputs = self.dinputs / samples

        return self.dinputs


class SoftmaxCategoricalCrossentropy(Loss):

  def __init__(self):
    self.activation = Softmax()
    self.loss = CategoricalCrossEntropy()

  def forward(self, inputs, y_true):
    self.activation.forward(inputs)

    self.output = self.activation.output

    return self.loss.calculate(self.output, y_true)

  def backward(self, dvalues, y_true):
  
    samples = len(dvalues)

    # turn into OHE
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)

    self.dinputs = dvalues.copy()

    self.dinputs[range(samples), y_true] -= 1 

    self.dinputs = self.dinputs / samples

    return self.dinputs