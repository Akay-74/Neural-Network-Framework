"""
Evaluation metrics for model performance
"""
import random

class Accuracy:
    """Classification accuracy metric"""
    def score(self, y_pred, y_true):
        """
        Assumes y_pred and y_true are Tensors (n_classes, 1)
        """
        correct = 0
        # Find index of max value for prediction
        pred_val = max(range(y_pred.rows), key=lambda i: y_pred.data[i][0])
        # Find index of 1 for true value
        true_val = max(range(y_true.rows), key=lambda i: y_true.data[i][0])

        if pred_val == true_val:
            correct = 1
        return correct # Trainer should sum this and divide by total


class Precision:
    """Precision for binary classification (assumes (1,1) tensor)"""
    def score(self, y_pred, y_true):
        tp = fp = 0
        pred_value = 1 if y_pred.data[0][0] >= 0.5 else 0
        true_value = int(y_true.data[0][0])
        if pred_value == 1:
            if true_value == 1:
                tp = 1
            else:
                fp = 1
        # This will be summed by a trainer
        return tp, fp


class Recall:
    """Recall for binary classification (assumes (1,1) tensor)"""
    def score(self, y_pred, y_true):
        tp = fn = 0
        pred_value = 1 if y_pred.data[0][0] >= 0.5 else 0
        true_value = int(y_true.data[0][0])
        if true_value == 1:
            if pred_value == 1:
                tp = 1
            else:
                fn = 1
        # This will be summed by a trainer
        return tp, fn


class F1Score:
    """F1 Score (harmonic mean of precision and recall)
    Note: This metric is stateful and complex to implement sample-by-sample.
    It's easier to compute from the final confusion matrix.
    This implementation is a placeholder.
    """
    def score(self, y_pred, y_true):
        print("Warning: F1Score is not accurately computed sample-by-sample.")
        return 0.0

