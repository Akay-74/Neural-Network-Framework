"""
Neural network model definition
"""
from .layers import Sequential, Layer

class Model:
    """Neural network model wrapper"""
    def __init__(self, layers):
        if not isinstance(layers, list) or not layers:
            raise ValueError("Model must be initialized with a non-empty list of layers.")
        if not all(isinstance(layer, Layer) for layer in layers):
             raise TypeError("All items in the layers list must be instances of the Layer class.")
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network.forward(x)

    def backward(self, grad, lr):
        return self.network.backward(grad, lr)
