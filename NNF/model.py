"""
Neural network model definition with optimizer support
"""
from .layers import Sequential, Layer


class Model:
    def __init__(self, layers):
        if not isinstance(layers, list) or not layers:
            raise ValueError("Model must be initialized with a non-empty list of layers.")
        if not all(isinstance(layer, Layer) for layer in layers):
            raise TypeError("All items in the layers list must be instances of the Layer class.")
        self.network = Sequential(*layers)
    def forward(self, x):
        return self.network.forward(x)

    def backward(self, grad):
        return self.network.backward(grad)
    
    def get_params(self):
        """Get all trainable parameters from the model"""
        return self.network.get_params()
    
    def set_params(self, params):
        """Set updated parameters in the model"""
        self.network.set_params(params)

