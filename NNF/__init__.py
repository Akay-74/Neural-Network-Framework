"""
NNF (Neural Network Framework) - A lightweight neural network library built with pure Python
"""

# Core components
from .matrix import Matrix
from .layers import Layer, Linear, ReLU, Sigmoid, Tanh, Sequential
from .losses import MSELoss
from .model import Model
from .trainer import Trainer
from .dataset import Dataset
from .metrics import Accuracy, Precision, Recall, F1Score
from .utils import MathUtils

__version__ = "1.0.0"
__author__ = "Your Name"

# Convenience imports for common usage patterns
__all__ = [
    # Core matrix operations
    'Matrix',
    
    # Layers
    'Layer', 'Linear', 'ReLU', 'Sigmoid', 'Tanh', 'Sequential',
    
    # Loss functions
    'MSELoss',
    
    # Model and training
    'Model', 'Trainer', 'Dataset',
    
    # Metrics
    'Accuracy', 'Precision', 'Recall', 'F1Score',
    
    # Utils
    'MathUtils'
]