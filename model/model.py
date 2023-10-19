from model.layers import Layer
from model.activations import Activation
from model.loss import Loss
from model.optimizer import Optimezer
from model.metrics import Metric

import numpy as np

class Model():

    def __init__(self):
        self.layers = []
        

    def add(self, layer: 'Layer', activation: 'Activation'):
        self.layers.append((layer, activation))
    
    def compile(self, loss: 'Loss', optimizer: 'Optimezer', metric: 'Metric'):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric

    def forward(self, X):
        for layer, activation in self.layers:
            layer.forward(X)
            X = layer.output
            if activation is not None:
                activation.forward(X)
                X = activation.output
        return X
    
    def backward(self, y):

        y = self.loss.backward(self.loss.output, y)

        for layer, activation in reversed(self.layers):
            if activation is not None:
                y = activation.backward(y)
            y = layer.backward(y)

    def fit(self, X, y, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size
        self.X = X
        self.y = y

        for epoch in range(epochs):
            epoch_loss = 0

            epoch_output = []

            for batch in range(0, len(X), batch_size):

                X_batch = X[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]

                output = self.forward(X_batch)
                epoch_output.extend(output)
                epoch_loss += self.loss.calculate(output, y_batch)

                self.backward(y_batch)
                self.optimizer.step(self.layers)

            if epoch % 100 == 0:
                self.metric.calculate(epoch_output, y)
                print(f'epoch: {epoch}/{epochs}, loss: {epoch_loss/len(X)}, {self.metric}')
