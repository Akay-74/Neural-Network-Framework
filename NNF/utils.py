"""
Utility functions for mathematical operations
"""


import math

class MathUtils:
    @staticmethod
    def exp(x, terms=20):
        """Approximate exponential using Taylor (Maclaurin) series"""
        result = 1.0
        term = 1.0
        for n in range(1, terms):
            term *= x / n
            result += term
        return result

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + MathUtils.exp(-x))

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation"""
        # tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        ex = MathUtils.exp(x)
        enx = MathUtils.exp(-x)
        return (ex - enx) / (ex + enx)

    @staticmethod
    def relu(x):
        """ReLU activation"""
        return max(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU activation"""
        return x if x > 0 else alpha * x

    @staticmethod
    def softmax(values):
        """Softmax activation (for a list of values)"""
        exp_vals = [MathUtils.exp(v) for v in values]
        total = sum(exp_vals)
        return [val / total for val in exp_vals]
