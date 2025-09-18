"""
Training utilities for neural networks
"""

class Trainer:
    """Training loop manager"""
    def __init__(self, model, loss_fn, lr=0.01, epochs=100):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataset, verbose=True):
        """Train the model on the given dataset"""
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(dataset)):
                x, y_true = dataset[i]

                # Forward pass
                y_pred = self.model.forward(x)
                total_loss += self.loss_fn.forward(y_pred, y_true)

                # Backward pass
                grad_loss = self.loss_fn.backward()
                self.model.backward(grad_loss, self.lr)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss={total_loss/len(dataset)}")

    def predict(self, X):
        """Make predictions on input data"""
        return [self.model.forward(x) for x in X]

    def evaluate(self, dataset):
        """Evaluate the model on a dataset"""
        total_loss = 0
        for i in range(len(dataset)):
            x, y_true = dataset[i]
            y_pred = self.model.forward(x)
            total_loss += self.loss_fn.forward(y_pred, y_true)
        return total_loss / len(dataset)