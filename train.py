import nnfs
from nnfs.datasets import spiral_data


from model.model import Model
from model.layers import Dense
from model.activations import ReLU
from model.loss import SoftmaxCategoricalCrossentropy
from model.metrics import Accuracy
from model.optimizer import SGD


nnfs.init()

# dataset
X, y = spiral_data(samples=100, classes=3)

model = Model()

model.add(Dense(2, 64), ReLU())
model.add(Dense(64, 3), None)

model.compile(SoftmaxCategoricalCrossentropy(), SGD(0.05), Accuracy())

model.fit(X, y, 10_000, 64)