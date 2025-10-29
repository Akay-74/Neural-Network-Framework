"""
Activation Functions (separated from layers like TensorFlow)
FIXED VERSION with proper Softmax implementation
"""
from .tensor import Tensor
from .utils import MathUtils
from .layers import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation"""
    def forward(self, x):
        self.x = x
        return x.apply(lambda v: max(0, v))

    def backward(self, grad_output):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else 0)


class Sigmoid(Layer):
    """Sigmoid activation function"""
    def forward(self, x):
        self.out = x.apply(lambda v: MathUtils.sigmoid(v))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out.apply(lambda s: s * (1 - s))


class Tanh(Layer):
    """Hyperbolic tangent activation"""
    def forward(self, x):
        self.out = x.apply(lambda v: MathUtils.tanh(v))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out.apply(lambda t: 1 - t * t)


class LeakyReLU(Layer):
    """Leaky ReLU activation with configurable slope"""
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v if v > 0 else self.alpha * v)

    def backward(self, grad_output):
        return grad_output * self.x.apply(lambda v: 1 if v > 0 else self.alpha)


class ELU(Layer):
    """Exponential Linear Unit activation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x.apply(lambda v: v if v > 0 else self.alpha * (MathUtils.exp(v) - 1))

    def backward(self, grad_output):
        return grad_output * self.x.apply(
            lambda v: 1 if v > 0 else self.alpha * MathUtils.exp(v)
        )


class Swish(Layer):
    """Swish activation (also known as SiLU)"""
    def forward(self, x):
        self.x = x
        self.sigmoid_x = x.apply(lambda v: MathUtils.sigmoid(v))
        return x * self.sigmoid_x

    def backward(self, grad_output):
        # Swish derivative: swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
        swish_x = self.x * self.sigmoid_x
        return grad_output * (swish_x + self.sigmoid_x * (swish_x.apply(lambda s: 1 - s)))


class GELU(Layer):
    """Gaussian Error Linear Unit activation"""
    def forward(self, x):
        self.x = x
        sqrt_2_over_pi = 0.7978845608
        return x.apply(lambda v: 0.5 * v * (1 + MathUtils.tanh(
            sqrt_2_over_pi * (v + 0.044715 * v ** 3)
        )))

    def backward(self, grad_output):
        sqrt_2_over_pi = 0.7978845608
        
        def gelu_derivative(v):
            tanh_arg = sqrt_2_over_pi * (v + 0.044715 * v ** 3)
            tanh_val = MathUtils.tanh(tanh_arg)
            sech2 = 1 - tanh_val * tanh_val
            return 0.5 * (1 + tanh_val) + 0.5 * v * sech2 * sqrt_2_over_pi * (1 + 3 * 0.044715 * v ** 2)
        
        return grad_output * self.x.apply(gelu_derivative)


class Softmax(Layer):
    """
    Softmax activation for multi-class classification
    FIXED: Properly handles column vectors and edge cases
    """
    def forward(self, x):
        """Applies softmax activation (expects column vector)."""
        self.x = x
        
        # Validate input shape
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError(f"Softmax expects column vector with shape (n, 1), got {x.shape}")
        
        # Extract values from column vector
        values = [row[0] for row in x.data]
        n = len(values)
        
        if n == 0:
            raise ValueError("Cannot apply softmax to empty tensor")
        
        # For numerical stability, subtract max
        max_val = max(values)
        
        # Compute exp(x - max) for all values
        exp_vals = [MathUtils.exp(v - max_val) for v in values]
        
        # Sum of all exp values
        sum_exp = sum(exp_vals)
        
        # Prevent division by zero
        if sum_exp == 0 or sum_exp < 1e-10:
            sum_exp = 1e-10
        
        # Normalize to get probabilities
        softmax_vals = [v / sum_exp for v in exp_vals]
        
        # Convert back to column vector Tensor
        self.out = Tensor([[v] for v in softmax_vals])
        
        return self.out

    def backward(self, grad_output):
        """
        Compute gradient of softmax
        FIXED: Proper Jacobian computation for softmax
        """
        n = len(self.out.data)
        if n == 0:
            return Tensor([])
            
        if grad_output.shape != (n, 1):
             raise ValueError(f"Gradient shape mismatch for Softmax. Expected ({n}, 1), but got {grad_output.shape}")

        grad_input = []
        
        # For each output dimension
        for i in range(n):
            si = self.out.data[i][0]
            grad_sum = 0.0
            
            # Compute sum over all j (Jacobian multiplication)
            for j in range(n):
                sj = self.out.data[j][0]
                grad_j = grad_output.data[j][0]
                
                if i == j:
                    # Diagonal elements: si * (1 - si)
                    grad_sum += grad_j * si * (1 - si)
                else:
                    # Off-diagonal elements: -si * sj
                    grad_sum += grad_j * (-si * sj)
            
            grad_input.append([grad_sum])
        
        return Tensor(grad_input)

