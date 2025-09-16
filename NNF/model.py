"""
Neural network model definition
"""
from .layers import Sequential


class Model:
    """Neural network model wrapper"""
    def __init__(self, layers):
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network.forward(x)

    def backward(self, grad, lr):
        return self.network.backward(grad, lr)