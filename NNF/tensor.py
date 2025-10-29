# tensor.py
"""
Changed Matrix.py to Tensor.py bcuz of its additional functionality
Supports scalars, lists.
FIXED: _get_shape logic to correctly handle empty lists.
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
        if not isinstance(data, list):
            return data
        # Check if all elements are non-lists (base case)
        if all(not isinstance(x, list) for x in data):
            if all(isinstance(x, (int, float)) for x in data):
                return data
            else:
                raise TypeError(f"Mixed types not allowed in tensor row: {data}")
        # Recurse if elements are lists
        elif all(isinstance(x, list) for x in data):
            return [self._convert(x) for x in data]
        else:
            raise TypeError(f"Mixed list and non-list elements in tensor: {data}")

    def _get_shape(self, data):
        """
        Robustly get shape of nested list structure.
        FIXED to handle empty lists correctly.
        """
        if not isinstance(data, list):
            return ()  # Scalar
        
        if len(data) == 0:
            return (0,) # Empty list
        
        # Get shape of the first element to determine subsequent dimensions
        first_shape = self._get_shape(data[0])
        
        # (Optional but good practice) Check for consistent shapes
        # for item in data[1:]:
        #     if self._get_shape(item) != first_shape:
        #         raise ValueError("Inconsistent shapes in tensor data")
                 
        return (len(data),) + first_shape

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
        elif len(self.shape) == 1: # e.g., shape (4,) is a vector
            return 1
        return 1 # e.g., shape () is a scalar

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    # ---------------- Element-wise operations ----------------
    def _elem_op(self, a, b, op):
        if isinstance(a, list) and isinstance(b, list):
            # Ensure shapes match for list-list operations
            if len(a) != len(b):
                raise ValueError(f"Shape mismatch for element-wise op: {len(a)} vs {len(b)}")
            return [self._elem_op(x, y, op) for x, y in zip(a, b)]
        elif isinstance(a, list): # a is list, b is scalar
            return [self._elem_op(x, b, op) for x in a]
        elif isinstance(b, list): # a is scalar, b is list
            return [self._elem_op(a, y, op) for y in b]
        else: # a is scalar, b is scalar
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
            def safe_div(a, b):
                if b == 0:
                    return float('inf') if a > 0 else -float('inf') if a < 0 else float('nan')
                return a / b
            return Tensor(self._elem_op(self.data, other.data, safe_div))
        else:
            if other == 0:
                 return Tensor(self._elem_op(self.data, other, lambda a,b: float('inf') if a > 0 else -float('inf') if a < 0 else float('nan')))
            return Tensor(self._elem_op(self.data, other, lambda a,b: a/b))

    # ---------------- Dot and matrix operations ----------------
    def dot(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"Dot product requires a Tensor, got {type(other)}")

        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape != other.shape:
                raise ValueError(f"Vectors must have same shape for dot product: {self.shape} vs {other.shape}")
            return sum(a*b for a,b in zip(self.data, other.data))
        
        elif len(self.shape) == 2 and len(other.shape) == 2:
            # W.dot(x) -> W=(out, in), x=(in, 1)
            # self.shape[1] is 'in'
            # other.shape[0] is 'in'
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} @ {other.shape}")
            
            result = []
            for i in range(self.shape[0]): # Iterate 'out'
                row = []
                for j in range(other.shape[1]): # Iterate '1'
                    s = sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1])) # Sum over 'in'
                    row.append(s)
                result.append(row)
            return Tensor(result) # Result shape (out, 1)
        
        else:
            raise NotImplementedError(f"Dot not implemented for shapes {self.shape} and {other.shape}")

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError(f"Transpose only for 2D tensors, got shape {self.shape}")
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
                raise ValueError(f"Axis 0 sum only implemented for 2D, got shape {self.shape}")
            return Tensor([sum(self.data[i][j] for i in range(self.shape[0])) for j in range(self.shape[1])])
        elif axis == 1:
            if len(self.shape) != 2:
                raise ValueError(f"Axis 1 sum only implemented for 2D, got shape {self.shape}")
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

