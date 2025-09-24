"""
NNF (Neural Network Framework) - A lightweight neural network library built with pure Python
"""

# Core components
from .tensor import Tensor
from .layers import Layer, Linear, ReLU, Sigmoid, Tanh, Sequential, LeakyReLU, ELU, Swish, GELU
from .losses import MSELoss, MAELoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from .model import Model
from .trainer import Trainer
from .metrics import Accuracy, Precision, Recall, F1Score
from .utils import MathUtils

__version__ = "1.0.0"
__author__ = "Aayaan, Shreyash"

# Convenience imports for common usage patterns
__all__ = [
    # Core tensor operations
    'Tensor',
    
    # Layers
    'Layer', 'Linear', 'ReLU', 'Sigmoid', 'Tanh', 'Sequential', 
    'LeakyReLU', 'ELU', 'Swish', 'GELU',
    
    # Loss functions
    'MSELoss', 'MAELoss', 'BinaryCrossEntropyLoss', 'CrossEntropyLoss',
    
    # Model and training
    'Model', 'Trainer',
    
    # Metrics
    'Accuracy', 'Precision', 'Recall', 'F1Score',
    
    # Utils
    'MathUtils'
]