"""
Loss functions for neural networks
"""
from .tensor import Tensor
from .utils import MathUtils

class _Loss:
    """Base class for loss functions to share validation logic."""
    def _validate_inputs(self, y_pred, y_true):
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError(f"Inputs to loss function must be Tensors, but got {type(y_pred)} and {type(y_true)}.")
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch between prediction ({y_pred.shape}) and true value ({y_true.shape}).")

class MSELoss(_Loss):
    """Mean Squared Error loss function"""
    def forward(self, y_pred, y_true):
        self._validate_inputs(y_pred, y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        
        n_elements = y_pred.rows * y_pred.cols
        if n_elements == 0:
            return 0.0
            
        diff = y_pred - y_true
        squared_diff = diff * diff
        return squared_diff.sum() / n_elements

    def backward(self):
        n_elements = self.y_pred.rows * self.y_pred.cols
        if n_elements == 0:
            return Tensor([])
            
        return (self.y_pred - self.y_true) * (2 / n_elements)

class MAELoss(_Loss):
    """Mean Absolute Error loss (L1 loss)"""
    def forward(self, y_pred, y_true):
        self._validate_inputs(y_pred, y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        
        n_elements = y_pred.rows * y_pred.cols
        if n_elements == 0:
            return 0.0
        
        diff = y_pred - y_true
        return diff.apply(MathUtils.abs).sum() / n_elements

    def backward(self):
        n_elements = self.y_pred.rows * self.y_pred.cols
        if n_elements == 0:
            return Tensor([])
            
        return (self.y_pred - self.y_true).apply(lambda x: 1 if x > 0 else -1) / n_elements

class BinaryCrossEntropyLoss(_Loss):
    """Binary Cross Entropy loss (for binary classification)"""
    def forward(self, y_pred, y_true):
        self._validate_inputs(y_pred, y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        
        n_elements = y_pred.rows * y_pred.cols
        if n_elements == 0:
            return 0.0
        
        eps = 1e-9
        y_pred_clipped = y_pred.apply(lambda x: MathUtils.clip(x, eps, 1 - eps))
        
        term1 = y_true * y_pred_clipped.apply(MathUtils.log)
        term2 = (y_true.apply(lambda x: 1-x)) * (y_pred_clipped.apply(lambda x: 1-x)).apply(MathUtils.log)
        
        loss_tensor = (term1 + term2) * -1
        return loss_tensor.sum() / n_elements

    def backward(self):
        n_elements = self.y_pred.rows * self.y_pred.cols
        if n_elements == 0:
            return Tensor([])
            
        eps = 1e-9
        y_pred_clipped = self.y_pred.apply(lambda x: MathUtils.clip(x, eps, 1 - eps))
        
        # Avoid division by zero
        denominator = y_pred_clipped * y_pred_clipped.apply(lambda x: 1 - x)
        denominator_clipped = denominator.apply(lambda x: eps if x == 0 else x)

        return ((y_pred_clipped - self.y_true) / denominator_clipped) / n_elements

class CrossEntropyLoss(_Loss):
    """Cross Entropy loss (for multi-class classification with softmax)"""
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (n_classes, 1) Tensor of probabilities from Softmax
            y_true: (n_classes, 1) Tensor (one-hot encoded)
        """
        self._validate_inputs(y_pred, y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        
        n_samples = y_pred.shape[0] # n_classes
        if n_samples == 0:
            return 0.0

        eps = 1e-9
        loss = 0.0
        # Assumes y_true is one-hot encoded (n_classes, 1)
        for i in range(n_samples):
            if y_true.data[i][0] == 1:
                clipped_pred = MathUtils.clip(y_pred.data[i][0], eps, 1.0)
                loss = -MathUtils.log(clipped_pred)
                break # Found the correct class
        
        return loss # Loss is for a single sample, trainer will average

    def backward(self):
        n_samples = self.y_pred.shape[0] # n_classes
        if n_samples == 0:
            return Tensor([])
        # This is the simplified gradient for (softmax output + cross-entropy loss)
        # Gradient is y_pred - y_true
        # The trainer will average, so we return the gradient for the single sample
        return (self.y_pred - self.y_true)

