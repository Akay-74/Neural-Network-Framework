"""
Training utilities for neural networks
"""
import random

class Trainer:
    """Training loop manager"""
    def __init__(self, model, loss_fn, lr=0.01, epochs=100):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, Y, verbose=True):
        """
        Train the model on dataset X, Y
        X, Y: lists of Tensors
        """
        n_samples = len(X)
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for x, y_true in zip(X, Y):
                # Forward pass
                y_pred = self.model.forward(x)
                loss = self.loss_fn.forward(y_pred, y_true)
                total_loss += loss

                # Backward pass
                grad_loss = self.loss_fn.backward()
                self.model.backward(grad_loss, self.lr)

            avg_loss = total_loss / n_samples
            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.epochs):
                print(f"Epoch {epoch}/{self.epochs} | Loss: {avg_loss:.6f}")

    def predict(self, X):
        """Return predictions for input list of Tensors"""
        return [self.model.forward(x) for x in X]

    def evaluate(self, X, Y):
        """Compute average loss on dataset"""
        total_loss = 0
        n_samples = len(X)
        for x, y_true in zip(X, Y):
            y_pred = self.model.forward(x)
            total_loss += self.loss_fn.forward(y_pred, y_true)
        return total_loss / n_samples