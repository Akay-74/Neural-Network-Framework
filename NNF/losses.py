"""
Loss functions for neural networks
"""
from .matrix import Matrix


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