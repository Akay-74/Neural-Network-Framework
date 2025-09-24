"""
Neural Network Layers
"""
from .tensor import Tensor
from .utils import MathUtils
import random

class Layer:
    """Base class for all layers"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output, lr=0.01):
        raise NotImplementedError


class Linear(Layer):
    """Fully connected layer"""
    def _init_(self, in_features, out_features):
        # Initialize weights with proper Xavier initialization
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.W = Tensor([[random.uniform(-limit, limit) for _ in range(in_features)] for _ in range(out_features)])
        self.b = Tensor([[0.0] for _ in range(out_features)])

    def forward(self, x):
        self.x = x
        return self.W.dot(x) + self.b

    def backward(self, grad_output, lr=0.01):
        # Compute gradients
        dW = grad_output.dot(self.x.transpose())
        db = grad_output
        
        # Update weights and biases
        self.W = self.W - dW * lr
        self.b = self.b - db * lr
        
        # Return gradient for previous layer
        return self.W.transpose().dot(grad_output)


class ReLU(Layer):
    def forward(self, x):
        self.x = x
        return x.apply(lambda v: max(0, v))

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else 0)


class Sigmoid(Layer):
    def forward(self, x):
        self.out = x.apply(lambda v: MathUtils.sigmoid(v))
        return self.out

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.out.apply(lambda s: s * (1 - s))


class Tanh(Layer):
    def forward(self, x):
        self.out = x.apply(lambda v: MathUtils.tanh(v))
        return self.out

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.out.apply(lambda t: 1 - t * t)


class LeakyReLU(Layer):
    def _init_(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v if v > 0 else self.alpha * v)

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else self.alpha)


class ELU(Layer):
    def _init_(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v if v > 0 else self.alpha * (MathUtils.exp(v) - 1))

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else self.alpha * MathUtils.exp(v))


class Swish(Layer):
    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v * MathUtils.sigmoid(v))

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.x.apply(lambda v: MathUtils.sigmoid(v) + v * MathUtils.sigmoid(v) * (1 - MathUtils.sigmoid(v)))


class GELU(Layer):
    def forward(self, x):
        self.x = x
        sqrt_2_over_pi = 0.7978845608
        return x.apply(lambda v: 0.5 * v * (1 + MathUtils.tanh(sqrt_2_over_pi * (v + 0.044715 * v ** 3))))

    def backward(self, grad_output, lr=0.01):
        sqrt_2_over_pi = 0.7978845608
        def gelu_derivative(v):
            tanh_val = MathUtils.tanh(sqrt_2_over_pi * (v + 0.044715 * v ** 3))
            sech2 = 1 - tanh_val * tanh_val
            return 0.5 * (1 + tanh_val) + 0.5 * v * sech2 * sqrt_2_over_pi * (1 + 3 * 0.044715 * v ** 2)
        return grad_output * self.x.apply(gelu_derivative)


class Sequential(Layer):
    def _init_(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, lr=0.01):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
        return grad_output