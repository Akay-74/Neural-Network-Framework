import random

class MathUtils:
    @staticmethod
    def exp(x=1, terms=20):
        """Approximate exponential using Taylor (Maclaurin) series"""
        if not isinstance(terms, int) or terms <= 0:
            raise ValueError("Number of terms for exp approximation must be a positive integer.")
        if x < -700: # Avoid underflow for large negative numbers
            return 0.0
        if x > 700: # Avoid overflow
            raise OverflowError("Input too large for exp approximation.")
            
        if x < 0:
            return 1.0 / MathUtils.exp(-x, terms)

        result = 1.0
        term = 1.0
        for n in range(1, terms):
            term *= x / n
            result += term
        return result

    @staticmethod
    def log(x, base=None, max_iter=30, tol=1e-9):
        """Natural log (or log base n) using Newton-Raphson and exp"""
        if x <= 0:
            raise ValueError("Logarithm is undefined for non-positive values.")
        if x == 1:
            return 0.0

        y = x - 1.0
        for _ in range(max_iter):
            try:
                e_y = MathUtils.exp(y)
                if e_y == 0: # Avoid division by zero
                    raise ValueError("Cannot compute log, exponential of guess is zero.")
                diff = e_y - x
                y -= diff / e_y
                if MathUtils.abs(diff) < tol:
                    break
            except OverflowError:
                 raise OverflowError("Overflow encountered during log calculation; input may be too large.")

        if base:
            if base <= 0 or base == 1:
                raise ValueError("Logarithm base must be positive and not equal to 1.")
            return y / MathUtils.log(base)
        return y

    @staticmethod
    def abs(x):
        """Absolute value"""
        return x if x >= 0 else -x

    @staticmethod
    def clip(x, min_val, max_val):
        """Clip value between min and max"""
        if min_val > max_val:
            raise ValueError("min_val cannot be greater than max_val in clip.")
        return max(min_val, min(x, max_val))

    @staticmethod
    def sigmoid(x):
        try:
            return 1 / (1 + MathUtils.exp(-x))
        except OverflowError: # exp(-x) overflows for large negative x
            return 0.0

    @staticmethod
    def tanh(x):
        try:
            ex = MathUtils.exp(x)
            enx = MathUtils.exp(-x)
            return (ex - enx) / (ex + enx)
        except OverflowError: # exp(x) overflows for large x
            return 1.0 if x > 0 else -1.0


    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return x if x > 0 else alpha * x

    @staticmethod
    def softmax(values):
        if not isinstance(values, list):
            raise TypeError("Softmax input must be a list of numbers.")
        if not values:
            raise ValueError("Softmax input cannot be an empty list.")
        try:
            max_val = max(values)
            exp_vals = [MathUtils.exp(v - max_val) for v in values]
            total = sum(exp_vals)
            if total == 0:
                # This can happen with underflow. Return a uniform distribution.
                return [1.0 / len(values)] * len(values)
            return [val / total for val in exp_vals]
        except (TypeError, ValueError) as e:
            raise TypeError(f"Softmax requires a list of numbers. Encountered error: {e}")
