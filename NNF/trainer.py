"""
Training utilities for neural networks with optimizer support
"""
from .model import Model
from .losses import _Loss
from .optimizers import Optimizer, SGD
from .tensor import Tensor # Import Tensor for type checking


class Trainer:
    """Training loop manager with optimizer support"""
    def __init__(self, model, loss_fn, optimizer=None, epochs=100):
        if not isinstance(model, Model):
            raise TypeError(f"model must be an instance of Model, not {type(model)}.")
        if not isinstance(loss_fn, _Loss):
            raise TypeError(f"loss_fn must be a valid loss function instance, not {type(loss_fn)}.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer.")
        
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        
        # Use provided optimizer or default to SGD
        if optimizer is None:
            self.optimizer = SGD(lr=0.01)
        else:
            if not isinstance(optimizer, Optimizer):
                raise TypeError(f"optimizer must be an instance of Optimizer, not {type(optimizer)}.")
            self.optimizer = optimizer

    def _validate_dataset(self, X, Y):
        """Checks if the dataset is valid for training."""
        if not isinstance(X, list) or not isinstance(Y, list):
            raise TypeError("X and Y must be lists of Tensors.")
        if not X:
            raise ValueError("Input data X cannot be empty.")
        if len(X) != len(Y):
            raise ValueError(f"Mismatch in number of samples between X ({len(X)}) and Y ({len(Y)}).")
        if not all(isinstance(x, Tensor) for x in X) or not all(isinstance(y, Tensor) for y in Y):
             raise TypeError("All elements in X and Y must be Tensors.")

    def fit(self, X, Y, verbose=True):
        """Train the model on dataset X, Y using the optimizer."""
        self._validate_dataset(X, Y)
        n_samples = len(X)
        
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for i, (x, y_true) in enumerate(zip(X, Y)):
                try:
                    # Forward pass
                    y_pred = self.model.forward(x)
                    loss = self.loss_fn.forward(y_pred, y_true)
                    total_loss += loss

                    # Backward pass
                    grad_loss = self.loss_fn.backward()
                    self.model.backward(grad_loss)
                    
                    # Update parameters using optimizer
                    params = self.model.get_params()
                    updated_params = self.optimizer.step(params)
                    self.model.set_params(updated_params)
                    
                except (ValueError, TypeError, NotImplementedError) as e:
                    print(f"Stopping training due to an error at epoch {epoch}, sample {i} (x.shape={x.shape}, y.shape={y_true.shape}): {e}")
                    # Re-raise the exception to stop training
                    raise e

            avg_loss = total_loss / n_samples
            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch {epoch}/{self.epochs} | Loss: {avg_loss:.6f}")

    def predict(self, X):
        """Return predictions for an input list of Tensors."""
        if not isinstance(X, list) or not all(isinstance(x, Tensor) for x in X):
            raise TypeError("Input X for prediction must be a list of Tensors.")
        return [self.model.forward(x) for x in X]

    def evaluate(self, X, Y):
        """Compute average loss on a dataset."""
        self._validate_dataset(X, Y)
        total_loss = 0
        n_samples = len(X)
        for x, y_true in zip(X, Y):
            y_pred = self.model.forward(x)
            total_loss += self.loss_fn.forward(y_pred, y_true)
        
        if n_samples == 0:
            return 0.0
        return total_loss / n_samples

