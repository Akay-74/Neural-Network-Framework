import math
import random

class Matrix:
    """Matrix operations using only built-in Python"""
    
    @staticmethod
    def create(rows, cols, fill_value=0.0):
        """Create a matrix filled with a specific value"""
        return [[fill_value for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def zeros(rows, cols):
        """Create a zero matrix"""
        return Matrix.create(rows, cols, 0.0)
    
    @staticmethod
    def random_matrix(rows, cols, min_val=-1.0, max_val=1.0):
        """Create a matrix with random values"""
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(random.uniform(min_val, max_val))
            matrix.append(row)
        return matrix
    
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
    def add(a, b):
        """Element-wise matrix addition"""
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Matrix dimensions must match for addition")
        
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(a[0])):
                row.append(a[i][j] + b[i][j])
            result.append(row)
        return result
    
    @staticmethod
    def subtract(a, b):
        """Element-wise matrix subtraction"""
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Matrix dimensions must match for subtraction")
        
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(a[0])):
                row.append(a[i][j] - b[i][j])
            result.append(row)
        return result
    
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
    def transpose(matrix):
        """Matrix transpose"""
        if not matrix:
            return []
        
        rows, cols = len(matrix), len(matrix[0])
        result = Matrix.zeros(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                result[j][i] = matrix[i][j]
        
        return result
    
    @staticmethod
    def scalar_multiply(matrix, scalar):
        """Multiply matrix by scalar"""
        result = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[0])):
                row.append(matrix[i][j] * scalar)
            result.append(row)
        return result
    
    @staticmethod
    def print_matrix(matrix, precision=4):
        """Print matrix in readable format"""
        for row in matrix:
            print([round(val, precision) for val in row])