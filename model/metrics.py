import numpy as np

def accuracy(y_hat, y_true):

    predictions = np.argmax(y_hat, axis=1)
    
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    return np.mean(predictions == y_true)