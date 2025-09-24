"""
Evaluation metrics for model performance
"""
import random

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