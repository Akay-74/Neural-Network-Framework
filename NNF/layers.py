"""
Neural Network Layers
"""
from .matrix import Matrix
from .utils import MathUtils


class Layer:
    """Base class for all layers - Used Inheritance for multiple types of layers"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError


class Linear(Layer):
    """Fully connected (dense) layer"""
    def __init__(self, in_features, out_features):
        self.W = Matrix.random_matrix(out_features, in_features, -0.1, 0.1)
        self.b = Matrix(out_features, 1, 0.0)

    def forward(self, x):
        self.x = x
        return (self.W @ x) + self.b

    def backward(self, grad_output, lr=0.01):
        dW = grad_output @ Matrix.transpose_static(self.x)
        db = grad_output
        self.W = self.W - dW.scalar_multiply(lr)
        self.b = self.b - db.scalar_multiply(lr)
        return Matrix.transpose_static(self.W) @ grad_output


class ReLU(Layer):
    """Rectified Linear Unit activation function"""
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                out.data[i][j] = max(0, x.data[i][j])
        return out
    
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                grad_input.data[i][j] = grad_output.data[i][j] if self.x.data[i][j] > 0 else 0
        return grad_input


class Sigmoid(Layer):
    """Sigmoid activation function"""
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                e = MathUtils.exp(-x.data[i][j])
                out.data[i][j] = 1.0 / (1.0 + e)
        self.out = out
        return out

    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                s = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * s * (1 - s)
        return grad_input


class Tanh(Layer):
    """Hyperbolic tangent activation function"""
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                e_pos = MathUtils.exp(x.data[i][j])
                e_neg = MathUtils.exp(-x.data[i][j])
                out.data[i][j] = (e_pos - e_neg) / (e_pos + e_neg)
        self.out = out
        return out

    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                t = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * (1 - t * t)
        return grad_input


class Sequential(Layer):
    """Sequential container for layers"""
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, lr=0.01):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
        return grad_output