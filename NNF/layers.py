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
    
class LeakyReLU(Layer):
    """Leaky ReLU: small slope for negative inputs, normal ReLU for positives"""
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                # Positive values pass unchanged, negatives are scaled down
                if x.data[i][j] > 0:
                    out.data[i][j] = x.data[i][j]
                else:
                    out.data[i][j] = self.alpha * x.data[i][j]
        return out
    
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                # Gradient is 1 for positives, alpha for negatives
                if self.x.data[i][j] > 0:
                    grad_input.data[i][j] = grad_output.data[i][j]
                else:
                    grad_input.data[i][j] = grad_output.data[i][j] * self.alpha
        return grad_input


class ELU(Layer):
    """ELU: smooth negative slope using exponent, avoids dead neurons"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                # Positive: linear, Negative: exponential curve
                if x.data[i][j] > 0:
                    out.data[i][j] = x.data[i][j]
                else:
                    out.data[i][j] = self.alpha * (MathUtils.exp(x.data[i][j]) - 1)
        self.out = out
        return out
    
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                # Gradient is 1 for positives, alpha*exp(x) for negatives
                if self.x.data[i][j] > 0:
                    grad_input.data[i][j] = grad_output.data[i][j]
                else:
                    grad_input.data[i][j] = grad_output.data[i][j] * self.alpha * MathUtils.exp(self.x.data[i][j])
        return grad_input


class Swish(Layer):
    """Swish: smooth function, combines sigmoid with input"""
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                # Formula: x * sigmoid(x)
                sigmoid_val = 1.0 / (1.0 + MathUtils.exp(-x.data[i][j]))
                out.data[i][j] = x.data[i][j] * sigmoid_val
        self.out = out
        return out
    
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                # Derivative: sigmoid(x) + x*sigmoid(x)*(1 - sigmoid(x))
                x_val = self.x.data[i][j]
                sigmoid_val = 1.0 / (1.0 + MathUtils.exp(-x_val))
                swish_derivative = sigmoid_val + x_val * sigmoid_val * (1 - sigmoid_val)
                grad_input.data[i][j] = grad_output.data[i][j] * swish_derivative
        return grad_input


class GELU(Layer):
    """GELU: smooth curve between ReLU and sigmoid, often used in transformers"""
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                # Approximation: 0.5 * x * (1 + tanh(...))
                x_val = x.data[i][j]
                sqrt_2_over_pi = 0.7978845608
                tanh_input = sqrt_2_over_pi * (x_val + 0.044715 * x_val**3)
                
                # Tanh calculated with exponentials
                e_pos = MathUtils.exp(tanh_input)
                e_neg = MathUtils.exp(-tanh_input)
                tanh_val = (e_pos - e_neg) / (e_pos + e_neg)
                
                out.data[i][j] = 0.5 * x_val * (1 + tanh_val)
        self.out = out
        return out
    
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
