import random

class MathUtils:
    @staticmethod
    def exp(x=1, terms=20):
        """Approximate exponential using Taylor (Maclaurin) series"""
        if x < 0:
            return 1.0 / MathUtils.exp(-x, terms)
        if x > 10:
            k = int(x // 10) + 1
            return (MathUtils.exp(x / k, terms)) ** k

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
            raise ValueError("Log undefined for non-positive values")

        # Initial guess
        y = x - 1 if x > 0.5 else 0.0

        for _ in range(max_iter):
            e_y = MathUtils.exp(y)
            diff = e_y - x
            y -= diff / e_y
            if abs(diff) < tol:
                break

        if base:
            return y / MathUtils.log(base)  # change of base
        return y

    @staticmethod
    def abs(x):
        """Absolute value"""
        return x if x >= 0 else -x

    @staticmethod
    def clip(x, min_val, max_val):
        """Clip value between min and max"""
        if x < min_val:
            return min_val
        elif x > max_val:
            return max_val
        else:
            return x

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            return 1 / (1 + MathUtils.exp(-x))
        else:
            ex = MathUtils.exp(x)
            return ex / (1 + ex)

    @staticmethod
    def tanh(x):
        ex = MathUtils.exp(x)
        enx = MathUtils.exp(-x)
        return (ex - enx) / (ex + enx)

    @staticmethod
    def relu(x):
        return x if x > 0 else 0

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return x if x > 0 else alpha * x

    @staticmethod
    def softmax(values):
        max_val = max(values)
        shifted = [v - max_val for v in values]
        exp_vals = [MathUtils.exp(v) for v in shifted]
        total = sum(exp_vals)
        return [val / total for val in exp_vals]