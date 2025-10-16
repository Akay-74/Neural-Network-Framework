"""
Training utilities for neural networks
"""
from .model import Model
from .losses import _Loss

class Trainer:
    """Training loop manager"""
    def __init__(self, model, loss_fn, lr=0.01, epochs=100):
        if not isinstance(model, Model):
            raise TypeError(f"model must be an instance of Model, not {type(model)}.")
        if not isinstance(loss_fn, _Loss):
            raise TypeError(f"loss_fn must be a valid loss function instance, not {type(loss_fn)}.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("Learning rate must be a positive number.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Number of epochs must be a positive integer.")
            
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs

    def _validate_dataset(self, X, Y):
        """Checks if the dataset is valid for training."""
        if not isinstance(X, list) or not isinstance(Y, list):
            raise TypeError("X and Y must be lists of Tensors.")
        if not X:
            raise ValueError("Input data X cannot be empty.")
        if len(X) != len(Y):
            raise ValueError(f"Mismatch in number of samples between X ({len(X)}) and Y ({len(Y)}).")

    def fit(self, X, Y, verbose=True):
        """Train the model on dataset X, Y."""
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
                    self.model.backward(grad_loss, self.lr)
                except (ValueError, TypeError, NotImplementedError) as e:
                    print(f"Stopping training due to an error at epoch {epoch}, sample {i}: {e}")
                    return # Exit the training loop

            avg_loss = total_loss / n_samples
            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch {epoch}/{self.epochs} | Loss: {avg_loss:.6f}")

    def predict(self, X):
        """Return predictions for an input list of Tensors."""
        if not isinstance(X, list):
            raise TypeError("Input X for prediction must be a list.")
        return [self.model.forward(x) for x in X]

    def evaluate(self, X, Y):
        """Compute average loss on a dataset."""
        self._validate_dataset(X, Y)
        total_loss = 0
        n_samples = len(X)
        for x, y_true in zip(X, Y):
            y_pred = self.model.forward(x)
            total_loss += self.loss_fn.forward(y_pred, y_true)
        return total_loss / n_samples
