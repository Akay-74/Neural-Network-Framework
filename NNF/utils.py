"""
Utility functions for mathematical operations
"""


class MathUtils:
    @staticmethod
    def exp(x, terms=10):
        """Approximate exponential using Taylor (Maclaurin) series, when a = 0 in taylor series"""
        result = 1.0
        term = 1.0
        for n in range(1, terms):
            term *= x / n
            result += term
        return result