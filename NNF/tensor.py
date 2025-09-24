# tensor.py
"""
Changed Matrix.py to Tensor.py bcuz of its additional functionality
Supports scalars, lists, NumPy arrays, Pandas Series/DataFrames
"""

import random

class Tensor:
    def __init__(self, data):
        if isinstance(data, (int, float)):  # scalar
            self.data = data
            self.shape = ()

        elif isinstance(data, list):  # list input
            self.data = self._convert(data)
            self.shape = self._get_shape(self.data)

        else:
            raise TypeError(f"Unsupported type for Tensor: {type(data)}")

    def _convert(self, data):
        if all(isinstance(x, (int, float)) for x in data):
            return data
        elif all(isinstance(x, list) for x in data):
            return [self._convert(x) for x in data]
        else:
            raise TypeError("Mixed types not allowed in tensor")

    def _get_shape(self, data):
        if isinstance(data, list) and len(data) > 0:
            return (len(data),) + self._get_shape(data[0])
        else:
            return ()

    @property
    def rows(self):
        """Get number of rows for 2D tensors"""
        if len(self.shape) >= 1:
            return self.shape[0]
        return 1

    @property
    def cols(self):
        """Get number of columns for 2D tensors"""
        if len(self.shape) >= 2:
            return self.shape[1]
        elif len(self.shape) == 1:
            return 1
        return 1

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    # ---------------- Element-wise operations ----------------
    def _elem_op(self, a, b, op):
        if isinstance(a, list) and isinstance(b, list):
            return [self._elem_op(x, y, op) for x, y in zip(a, b)]
        elif isinstance(a, list):
            return [self._elem_op(x, b, op) for x in a]
        elif isinstance(b, list):
            return [self._elem_op(a, y, op) for y in b]
        else:
            return op(a, b)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elem_op(self.data, other.data, lambda a,b: a+b))
        else:
            return Tensor(self._elem_op(self.data, other, lambda a,b: a+b))

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elem_op(self.data, other.data, lambda a,b: a-b))
        else:
            return Tensor(self._elem_op(self.data, other, lambda a,b: a-b))

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elem_op(self.data, other.data, lambda a,b: a*b))
        else:
            return Tensor(self._elem_op(self.data, other, lambda a,b: a*b))

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elem_op(self.data, other.data, lambda a,b: a/b))
        else:
            return Tensor(self._elem_op(self.data, other, lambda a,b: a/b))

    # ---------------- Dot and matrix operations ----------------
    def dot(self, other):
        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape != other.shape:
                raise ValueError("Vectors must have same shape")
            return sum(a*b for a,b in zip(self.data, other.data))
        elif len(self.shape) == 2 and len(other.shape) == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError("Incompatible shapes for matrix multiplication")
            result = []
            for i in range(self.shape[0]):
                row = []
                for j in range(other.shape[1]):
                    s = sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1]))
                    row.append(s)
                result.append(row)
            return Tensor(result)
        else:
            raise NotImplementedError("Dot not implemented for >2D tensors")

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("Transpose only for 2D tensors")
        return Tensor([[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])])

    def apply(self, func):
        """Apply function element-wise to tensor"""
        def recursive_apply(data):
            if isinstance(data, list):
                return [recursive_apply(x) for x in data]
            else:
                return func(data)
        
        return Tensor(recursive_apply(self.data))

    # ---------------- Sum along axis ----------------
    def sum(self, axis=None):
        if axis is None:
            # sum all elements
            def recursive_sum(d):
                if isinstance(d, list):
                    return sum(recursive_sum(x) for x in d)
                else:
                    return d
            return recursive_sum(self.data)
        elif axis == 0:
            if len(self.shape) != 2:
                raise ValueError("Axis 0 sum only implemented for 2D")
            return Tensor([sum(self.data[i][j] for i in range(self.shape[0])) for j in range(self.shape[1])])
        elif axis == 1:
            if len(self.shape) != 2:
                raise ValueError("Axis 1 sum only implemented for 2D")
            return Tensor([sum(row) for row in self.data])
        else:
            raise ValueError("Axis >1 not implemented")
        
    def get_shape(self):
        return self.shape

    @staticmethod
    def random(rows, cols=None, low=0.0, high=1.0):
        """Generate a random Tensor. If cols=None, creates vector/scalar."""
        if cols is None:
            if isinstance(rows, int):
                return Tensor([random.uniform(low, high) for _ in range(rows)])
            elif isinstance(rows, tuple):
                def build(shape):
                    if len(shape) == 1:
                        return [random.uniform(low, high) for _ in range(shape[0])]
                    else:
                        return [build(shape[1:]) for _ in range(shape[0])]
                return Tensor(build(rows))
        else:
            return Tensor([[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)])

    def scalar_multiply(self, scalar):
        return self * scalar

    def __matmul__(self, other):
        """Matrix multiplication for 2D Tensors or vector dot product."""
        return self.dot(other)

    @staticmethod
    def transpose_static(tensor):
        return tensor.transpose()