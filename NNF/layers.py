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
    """Fully connected (dense) layer"""
    def __init__(self, in_features, out_features):
        self.W = Matrix.random_matrix(out_features, in_features, -0.1, 0.1)
        self.b = Matrix(out_features, 1, 0.0)

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
    """Leaky ReLU: small slope for negative inputs, normal ReLU for positives"""
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v if v > 0 else self.alpha * v)

    def backward(self, grad_output, lr=0.01):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else self.alpha)


class ELU(Layer):
    """ELU: smooth negative slope using exponent, avoids dead neurons"""
    def __init__(self, alpha=1.0):
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
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                # Approximate derivative of GELU
                x_val = self.x.data[i][j]
                sqrt_2_over_pi = 0.7978845608
                tanh_input = sqrt_2_over_pi * (x_val + 0.044715 * x_val**3)
                
                # Tanh and sech²
                e_pos = MathUtils.exp(tanh_input)
                e_neg = MathUtils.exp(-tanh_input)
                tanh_val = (e_pos - e_neg) / (e_pos + e_neg)
                sech_squared = 1 - tanh_val * tanh_val
                
                # Inner derivative term
                inner_derivative = sqrt_2_over_pi * (1 + 3 * 0.044715 * x_val**2)
                
                # Final derivative expression
                gelu_derivative = 0.5 * (1 + tanh_val) + 0.5 * x_val * sech_squared * inner_derivative
                
                grad_input.data[i][j] = grad_output.data[i][j] * gelu_derivative
        return grad_input
