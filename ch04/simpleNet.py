import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.grandient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 2 inputs, 3 outputs

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)
