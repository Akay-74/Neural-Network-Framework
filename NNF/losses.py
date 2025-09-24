"""
Loss functions for neural networks
"""
from .tensor import Tensor
from .utils import MathUtils
import random

class MSELoss:
    """Mean Squared Error loss function"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = sum(
            (y_pred.data[i][j] - y_true.data[i][j]) ** 2
            for i in range(y_pred.shape[0])
            for j in range(y_pred.shape[1])
        )
        return loss / (y_pred.shape[0] * y_pred.shape[1])

    def backward(self):
        return Tensor([
            [
                2 * (self.y_pred.data[i][j] - self.y_true.data[i][j]) / (self.y_pred.shape[0] * self.y_pred.shape[1])
                for j in range(self.y_pred.shape[1])
            ]
            for i in range(self.y_pred.shape[0])
        ])


class MAELoss:
    """Mean Absolute Error loss (L1 loss)"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = sum(
            MathUtils.abs(y_pred.data[i][j] - y_true.data[i][j])
            for i in range(y_pred.shape[0])
            for j in range(y_pred.shape[1])
        )
        return loss / (y_pred.shape[0] * y_pred.shape[1])

    def backward(self):
        return Tensor([
            [
                (1 if self.y_pred.data[i][j] - self.y_true.data[i][j] > 0 else -1) / (self.y_pred.shape[0] * self.y_pred.shape[1])
                for j in range(self.y_pred.shape[1])
            ]
            for i in range(self.y_pred.shape[0])
        ])


class BinaryCrossEntropyLoss:
    """Binary Cross Entropy loss (for binary classification)"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        eps = 1e-9
        loss = sum(
            -(y_true.data[i][j] * MathUtils.log(MathUtils.clip(y_pred.data[i][j], eps, 1 - eps)) +
              (1 - y_true.data[i][j]) * MathUtils.log(MathUtils.clip(1 - y_pred.data[i][j], eps, 1 - eps)))
            for i in range(y_pred.shape[0])
            for j in range(y_pred.shape[1])
        )
        return loss / (y_pred.shape[0] * y_pred.shape[1])

    def backward(self):
        eps = 1e-9
        return Tensor([
            [
                (MathUtils.clip(self.y_pred.data[i][j], eps, 1 - eps) - self.y_true.data[i][j]) /
                (MathUtils.clip(self.y_pred.data[i][j], eps, 1 - eps) *
                 MathUtils.clip(1 - self.y_pred.data[i][j], eps, 1 - eps) *
                 self.y_pred.shape[0] * self.y_pred.shape[1])
                for j in range(self.y_pred.shape[1])
            ]
            for i in range(self.y_pred.shape[0])
        ])


class CrossEntropyLoss:
    """Cross Entropy loss (for multi-class classification with softmax)"""
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        eps = 1e-9
        loss = sum(
            -MathUtils.log(MathUtils.clip(y_pred.data[i][j], eps, 1))
            for i in range(y_pred.shape[0])
            for j in range(y_pred.shape[1])
            if y_true.data[i][j] == 1
        )
        return loss / y_pred.shape[0]

    def backward(self):
        return Tensor([
            [
                (self.y_pred.data[i][j] - self.y_true.data[i][j]) / self.y_pred.shape[0]
                for j in range(self.y_pred.shape[1])
            ]
            for i in range(self.y_pred.shape[0])
        ])