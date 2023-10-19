import numpy as np

class Metric:

    def calculate(self, y_hat, y_true):
        pass

class Accuracy(Metric):

    def calculate(self, y_hat, y_true):
        
        predictions = np.argmax(y_hat, axis=1)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.results = np.mean(predictions == y_true)
        return self.results
    
    def __str__(self) -> str:
        return f'acc: {self.results}'