"""
Loss functions for neural networks
"""
from .matrix import Matrix
import math


class MSELoss:
    """Mean Squared Error loss function"""
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


class MAELoss:
    """Mean Absolute Error loss (L1 loss)"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = 0.0
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                loss += abs(y_pred.data[i][j] - y_true.data[i][j])
        return loss / (y_pred.rows * y_pred.cols)

    def backward(self):
        grad = Matrix(self.y_pred.rows, self.y_pred.cols)
        for i in range(self.y_pred.rows):
            for j in range(self.y_pred.cols):
                diff = self.y_pred.data[i][j] - self.y_true.data[i][j]
                grad.data[i][j] = (1 if diff > 0 else -1) / (self.y_pred.rows * self.y_pred.cols)
        return grad


class BinaryCrossEntropyLoss:
    """Binary Cross Entropy loss (for binary classification)"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = 0.0
        eps = 1e-9  # prevent log(0)
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                y = y_true.data[i][j]
                p = min(max(y_pred.data[i][j], eps), 1 - eps)  # clip to [eps,1-eps]
                loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return loss / (y_pred.rows * y_pred.cols)

    def backward(self):
        grad = Matrix(self.y_pred.rows, self.y_pred.cols)
        eps = 1e-9
        for i in range(self.y_pred.rows):
            for j in range(self.y_pred.cols):
                y = self.y_true.data[i][j]
                p = min(max(self.y_pred.data[i][j], eps), 1 - eps)
                grad.data[i][j] = (p - y) / (p * (1 - p) * self.y_pred.rows * self.y_pred.cols)
        return grad


class CrossEntropyLoss:
    """Cross Entropy loss (for multi-class classification with softmax)"""
    def forward(self, y_pred, y_true):
        """
        y_pred: probabilities (after softmax)
        y_true: one-hot encoded labels
        """
        self.y_pred = y_pred
        self.y_true = y_true
        loss = 0.0
        eps = 1e-9
        for i in range(y_pred.rows):
            for j in range(y_pred.cols):
                if y_true.data[i][j] == 1:
                    loss += -math.log(max(y_pred.data[i][j], eps))
        return loss / y_pred.rows

    def backward(self):
        grad = Matrix(self.y_pred.rows, self.y_pred.cols)
        for i in range(self.y_pred.rows):
            for j in range(self.y_pred.cols):
                grad.data[i][j] = (self.y_pred.data[i][j] - self.y_true.data[i][j]) / self.y_pred.rows
        return grad
