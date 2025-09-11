import random

class Matrix:
    """Matrix operations using only built-in Python"""

    def __init__(self, rows, cols, fill_value=0.0):#the constructor stores rows, cols and data itself too
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
    
    def __matmul__(self, other):  # makes @ operator to be used as a function for vector multiplication
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
    
    def __add__(self, other):  #used __add__ for + operator to be used as a function
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __sub__(self, other):  #used __sub__ for - operator to be used as a function 
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
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
    
    def __str__(self):#makes you able to print matrix using print(obj_name) function
        return "\n".join(str(row) for row in self.data)

"""Layers"""
class Layer:
    """Used Inheritance for multiple types of layers """
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = Matrix.random_matrix(out_features, in_features, -0.1, 0.1)
        self.b = Matrix(out_features, 1, 0.0)

    def forward(self, x):# performs linear transformation of input features and saves x for later use in backpropogation
        self.x = x
        return (self.W @ x) + self.b

    def backward(self, grad_output, lr=0.01):# backpropogation
        dW = grad_output @ self.x.transpose()
        db = grad_output
        self.W = self.W - dW.scalar_multiply(lr)# updating weights
        self.b = self.b - db.scalar_multiply(lr)# updating bias
        return self.W.transpose() @ grad_output
    
class ReLU(Layer):#ReLu activation function
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                out.data[i][j] = max(0, x.data[i][j])
        return out
    
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                grad_input.data[i][j] = grad_output.data[i][j] if self.x.data[i][j] > 0 else 0
        return grad_input

class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, lr=0.01):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
        return grad_output

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = 0.0
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                diff = y_pred.data[i][j] - y_true.data[i][j]
                loss += diff * diff
        return loss / (y_pred.rows * y_pred.cols)

    def backward(self):
        grad = Matrix(self.y_pred.rows, self.y_pred.cols)
        for i in range(self.y_pred.rows):
            for j in range(self.y_pred.cols):
                grad.data[i][j] = 2 * (self.y_pred.data[i][j] - self.y_true.data[i][j]) / (self.y_pred.rows * self.y_pred.cols)
        return grad

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

class Sigmoid(Layer):
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                e = MathUtils.exp(-x.data[i][j])
                out.data[i][j] = 1.0 / (1.0 + e)
        self.out = out
        return out

    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                s = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * s * (1 - s)

class Tanh(Layer):
    def forward(self, x):
        self.x = x
        out = Matrix(x.rows, x.cols)
        for i in range(x.rows):
            for j in range(x.cols):
                e_pos = MathUtils.exp(x.data[i][j])
                e_neg = MathUtils.exp(-x.data[i][j])
                out.data[i][j] = (e_pos - e_neg) / (e_pos + e_neg)
        self.out = out
        return out

    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                t = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * (1 - t * t)
        return grad_input
        return grad_input

class Dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class Model:
    def __init__(self, layers):
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network.forward(x)

    def backward(self, grad, lr):
        return self.network.backward(grad, lr)

class Trainer:
    def __init__(self, model, loss_fn, lr=0.01, epochs=100):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataset, verbose=True):
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(dataset)):
                x, y_true = dataset[i]

                # Forward
                y_pred = self.model.forward(x)
                total_loss += self.loss_fn.forward(y_pred, y_true)

                # Backward
                grad_loss = self.loss_fn.backward()
                self.model.backward(grad_loss, self.lr)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss={total_loss/len(dataset)}")

    def predict(self, X):
        return [self.model.forward(x) for x in X]

    def evaluate(self, dataset):
        total_loss = 0
        for i in range(len(dataset)):
            x, y_true = dataset[i]
            y_pred = self.model.forward(x)
            total_loss += self.loss_fn.forward(y_pred, y_true)
        return total_loss / len(dataset)

class Accuracy:
    """Classification accuracy metric"""
    def score(self, y_pred, y_true):
        correct = 0
        total = y_pred.rows * y_pred.cols
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                # Binary classification: round prediction to 0/1
                pred_value = 1 if y_pred.data[i][j] >= 0.5 else 0
                true_value = int(y_true.data[i][j])
                if pred_value == true_value:
                    correct += 1
        return correct / total


class Precision:
    """Precision for binary classification"""
    def score(self, y_pred, y_true):
        tp = fp = 0
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                pred_value = 1 if y_pred.data[i][j] >= 0.5 else 0
                true_value = int(y_true.data[i][j])
                if pred_value == 1:
                    if true_value == 1:
                        tp += 1
                    else:
                        fp += 1
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


class Recall:
    """Recall for binary classification"""
    def score(self, y_pred, y_true):
        tp = fn = 0
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                pred_value = 1 if y_pred.data[i][j] >= 0.5 else 0
                true_value = int(y_true.data[i][j])
                if true_value == 1:
                    if pred_value == 1:
                        tp += 1
                    else:
                        fn += 1
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class F1Score:
    """F1 Score (harmonic mean of precision and recall)"""
    def score(self, y_pred, y_true):
        precision_metric = Precision().score(y_pred, y_true)
        recall_metric = Recall().score(y_pred, y_true)
        if precision_metric + recall_metric == 0:
            return 0.0
        return 2 * (precision_metric * recall_metric) / (precision_metric + recall_metric)