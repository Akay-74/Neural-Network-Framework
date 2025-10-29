"""
NNF (Neural Network Framework) - A lightweight neural network library built with pure Python
FIXED VERSION: Properly exports Linear alias
"""

# Core components
from .tensor import Tensor
from .layers import Layer, Dense, Linear, Sequential  # Linear is an alias for Dense
from .activations import ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax
from .losses import MSELoss, MAELoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from .optimizers import SGD, Momentum, RMSprop, Adam
from .model import Model
from .trainer import Trainer
from .metrics import Accuracy, Precision, Recall, F1Score
# .utils is internal, not exported

__version__ = "1.1.0"
__author__ = "Aayaan, Shreyash"

# Convenience imports for common usage patterns
__all__ = [
    # Core tensor operations
    'Tensor',
    
    # Layers (FIXED: Added Linear to exports)
    'Layer', 'Dense', 'Linear', 'Sequential',
    
    # Activation functions
    'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'Swish', 'GELU', 'Softmax',
    
    # Loss functions
    'MSELoss', 'MAELoss', 'BinaryCrossEntropyLoss', 'CrossEntropyLoss',
    
    # Optimizers
    'SGD', 'Momentum', 'RMSprop', 'Adam',
    
    # Model and training
    'Model', 'Trainer',
    
    # Metrics
    'Accuracy', 'Precision', 'Recall', 'F1Score',
]

