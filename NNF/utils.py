class MathUtils:
    """Static math utility class"""
    
    @staticmethod
    def exp(x, terms=20):
        if x < 0:
            return 1.0 / MathUtils.exp(-x, terms)
        if x > 10:
            k = int(x // 10) + 1
            return MathUtils.exp(x / k, terms) ** k
        result = 1.0
        term = 1.0
        for n in range(1, terms):
            term *= x / n
            result += term
        return result

    @staticmethod
    def log(x, base=None, max_iter=30, tol=1e-9):
        if x <= 0:
            raise ValueError("Log undefined for non-positive values")
        y = x - 1 if x > 0.5 else 0.0
        for _ in range(max_iter):
            e_y = MathUtils.exp(y)
            diff = e_y - x
            y -= diff / e_y
            if abs(diff) < tol:
                break
        if base:
            return y / MathUtils.log(base)
        return y

    @staticmethod
    def clip(x, min_val, max_val):
        return max(min_val, min(x, max_val))


class Activation:
    """Base class for activation functions"""
    def __call__(self, x):
        return self.compute(x)
    
    def compute(self, x):
        raise NotImplementedError("Must implement compute method")


class Sigmoid(Activation):
    def compute(self, x):
        if x >= 0:
            return 1 / (1 + MathUtils.exp(-x))
        else:
            ex = MathUtils.exp(x)
            return ex / (1 + ex)


class Tanh(Activation):
    def compute(self, x):
        ex = MathUtils.exp(x)
        enx = MathUtils.exp(-x)
        return (ex - enx) / (ex + enx)


class ReLU(Activation):
    def compute(self, x):
        return x if x > 0 else 0


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def compute(self, x):
        return x if x > 0 else self.alpha * x


class Softmax(Activation):
    def compute(self, values):
        max_val = max(values)
        shifted = [v - max_val for v in values]
        exp_vals = [MathUtils.exp(v) for v in shifted]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]


# ---------------- Example usage ----------------
sigmoid = Sigmoid()
tanh = Tanh()
relu = ReLU()
leaky = LeakyReLU(0.05)
softmax = Softmax()

print(sigmoid(2))
print(tanh(1))
print(relu(-3))
print(leaky(-3))
print(softmax([1, 2, 3]))
