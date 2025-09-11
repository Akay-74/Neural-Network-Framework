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
    
    #Added missing indexing methods
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

    #Added scalar multiplication method for Matrix objects
    def scalar_multiply(self, scalar):
        """Multiply this matrix by a scalar"""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
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
        """Transpose a Matrix object"""
        if not matrix or not matrix.data:
            return Matrix(0, 0)
        
        rows, cols = matrix.rows, matrix.cols
        result = Matrix.zeros(cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                result.data[j][i] = matrix.data[i][j]
        
        return result
    
    #Added transpose method for Matrix instances
    def transpose(self):
        """Transpose this matrix"""
        result = Matrix.zeros(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
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
    
    def __str__(self):
        return "\n".join(str(row) for row in self.data)

"""Layers"""
class Layer:
    """Used Inheritance for multiple types of layers"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = Matrix.random_matrix(out_features, in_features, -0.1, 0.1)
        self.b = Matrix(out_features, 1, 0.0)

    def forward(self, x):
        self.x = x
        return (self.W @ x) + self.b

    # Fixed: Corrected backward method calls
    def backward(self, grad_output, lr=0.01):
        dW = grad_output @ Matrix.transpose(self.x)
        db = grad_output
        self.W = self.W - dW.scalar_multiply(lr)
        self.b = self.b - db.scalar_multiply(lr)
        return Matrix.transpose(self.W) @ grad_output
    
class ReLU(Layer):
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

    # Fixed: Added missing return statement
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                s = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * s * (1 - s)
        return grad_input

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

    # Fixed: Removed duplicate return statement
    def backward(self, grad_output, lr=0.01):
        grad_input = Matrix(self.x.rows, self.x.cols)
        for i in range(self.x.rows):
            for j in range(self.x.cols):
                t = self.out.data[i][j]
                grad_input.data[i][j] = grad_output.data[i][j] * (1 - t * t)
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

#Implementing created framework as of now
if __name__ == "__main__":
    X, Y = Matrix.create_dataset(samples=100, input_features=3, output_features=1)
    dataset = Dataset(X, Y)

    layers = [Linear(3, 5), ReLU(), Linear(5, 1), Sigmoid()]
    model = Model(layers)

    loss_fn = MSELoss()
    trainer = Trainer(model, loss_fn, lr=0.01, epochs=200)

    trainer.fit(dataset)

    # Evaluate total loss on the dataset
    total_loss = trainer.evaluate(dataset)
    print(f"\nFinal Loss on Dataset: {total_loss}\n")

    # Make predictions on first 5 samples
    print("Predictions vs True Values (first 5 samples):")
    y_pred = trainer.predict(X[:5])
    for i in range(5):
        print(f"Predicted: {y_pred[i].data} | True: {Y[i].data}")

    # Compute metrics
    accuracy = Accuracy().score(y_pred[0], Y[0])
    print(f"\nAccuracy (first sample): {accuracy}")