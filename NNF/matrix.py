"""
Matrix operations using only built-in Python
"""
import random


class Matrix:
    """Matrix operations using only built-in Python"""

    def __init__(self, rows, cols, fill_value=0.0):
        self.rows = rows
        self.cols = cols
        self.data = [[fill_value for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def zeros(rows, cols):
        """Create a zero matrix"""
        return Matrix(rows, cols, 0.0)
    
    @staticmethod
    def random_matrix(rows, cols, min_val=-1.0, max_val=1.0):
        """Create a matrix with random values"""
        m = Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                m.data[i][j] = random.uniform(min_val, max_val)
        return m
    
    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions do not align for multiplication")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0
                for k in range(self.cols):
                    s += self.data[i][k] * other.data[k][j]
                result.data[i][j] = s
        return result
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i][j]
        else:
            return self.data[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            self.data[i][j] = value
        else:
            self.data[key] = value

    def __add__(self, other):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __sub__(self, other):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def scalar_multiply(self, scalar):
        """Multiply this matrix by a scalar"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
        return result

    def transpose(self):
        """Transpose this matrix"""
        result = Matrix.zeros(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def __str__(self):
        return "\n".join(str(row) for row in self.data)

    @staticmethod
    def dot(a, b):
        """Matrix multiplication"""
        if not a or not b:
            raise ValueError("Empty matrices")
        
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply matrices: {cols_a} != {rows_b}")
        
        result = Matrix.zeros(rows_a, cols_b)
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    @staticmethod
    def create_dataset(samples=50, input_features=3, output_features=1, min_val=0.0, max_val=1.0):
        """Create a dataset of random matrices"""
        X, Y = [], []
        for _ in range(samples):
            x = Matrix.random_matrix(input_features, 1, min_val, max_val)
            y = Matrix.random_matrix(output_features, 1, min_val, max_val)
            X.append(x)
            Y.append(y)
        return X, Y
    
    @staticmethod
    def multiply(a, b):
        """Element-wise matrix multiplication"""
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Matrix dimensions must match for element-wise multiplication")
        
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(a[0])):
                row.append(a[i][j] * b[i][j])
            result.append(row)
        return result
    
    @staticmethod
    def transpose_static(matrix):
        """Transpose a Matrix object (static version)"""
        if not matrix or not matrix.data:
            return Matrix(0, 0)
        
        rows, cols = matrix.rows, matrix.cols
        result = Matrix.zeros(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                result.data[j][i] = matrix.data[i][j]
        
        return result
    
    @staticmethod
    def scalar_multiply_static(matrix, scalar):
        """Multiply matrix by scalar (static version)"""
        result = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[0])):
                row.append(matrix[i][j] * scalar)
            result.append(row)
        return result