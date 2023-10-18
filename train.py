import nnfs
from nnfs.datasets import spiral_data


from model.layers import Layer_Dense
from model.activations import ReLU
from model.loss import SoftmaxCategoricalCrossentropy

from model.metrics import accuracy

nnfs.init()

# dataset
X, y = spiral_data(samples=100, classes=3)


# model
dense1 = Layer_Dense(2, 64)
activation1 = ReLU()

dense2 = Layer_Dense(64, 3)
loss_activation = SoftmaxCategoricalCrossentropy()


for epoch in range(30_001):

    # forward
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    if not epoch % 1000:
        acc = accuracy(loss_activation.output, y)

        print(f'epcoh: {epoch} ' +
            f'acc: {acc} ' +
            f'loss: {loss}')

    # backpropagation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # SGD
    dense1.weights += -dense1.dweights
    dense1.biases += -dense1.dbiases

    dense2.weights += -dense2.dweights
    dense2.biases += -dense2.dbiases