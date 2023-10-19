

class Optimezer:

    def step(self):
        pass


class SGD(Optimezer):

    def __init__(self, lr=1.0):
        self.lr = lr

    def step(self, layers):
        for layer, _ in layers:
            layer.weights += -self.lr * layer.dweights
            layer.biases += -self.lr * layer.dbiases