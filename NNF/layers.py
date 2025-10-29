"""
Neural Network Layers (FIXED: Proper matrix multiplication dimensions)
This is the file that was likely cached. This version is correct.
"""
from .tensor import Tensor
import random

class Layer:
    """Base class for all layers"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params(self):
        """Return parameters that need to be updated by optimizer"""
        return []
    
    def set_params(self, params):
        """Set parameters from optimizer"""
        pass


class Dense(Layer):
    """Fully connected layer with proper dimensions
    
    For input x of shape (in_features, 1) and weights W of shape (out_features, in_features),
    the output is W @ x which gives shape (out_features, 1)
    """
    def __init__(self, in_features, out_features, use_bias=True):
        # Initialize weights: W has shape (out_features, in_features)
        # This is the line that *must* be correct.
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.W = Tensor([[random.uniform(-limit, limit) for _ in range(in_features)] 
                        for _ in range(out_features)])
        
        self.use_bias = use_bias
        if use_bias:
            # Bias b has shape (out_features, 1)
            self.b = Tensor([[0.0] for _ in range(out_features)])
        else:
            self.b = None
        
        # Store gradients for optimizer
        self.dW = None
        self.db = None
        
        # Store input shape for validation
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """Forward pass: output = W @ x + b
        
        Args:
            x: Tensor of shape (in_features, 1)
        
        Returns:
            Tensor of shape (out_features, 1)
        """
        if x.shape != (self.in_features, 1):
            raise ValueError(f"Input shape mismatch for Dense layer. Expected ({self.in_features}, 1), but got {x.shape}")

        self.x = x
        
        # W is (out_features, in_features)
        # x is (in_features, 1)
        # W.dot(x) gives (out_features, 1)
        output = self.W.dot(x)
        
        if self.use_bias:
            output = output + self.b
        return output

    def backward(self, grad_output):
        """Backward pass
        
        Args:
            grad_output: Tensor of shape (out_features, 1)
        
        Returns:
            Tensor of shape (in_features, 1) - gradient w.r.t. input
        """
        if grad_output.shape != (self.out_features, 1):
             raise ValueError(f"Gradient shape mismatch for Dense layer. Expected ({self.out_features}, 1), but got {grad_output.shape}")

        # Compute gradient w.r.t. weights: dW = grad_output @ x^T
        # grad_output is (out_features, 1), x^T is (1, in_features)
        # Result is (out_features, in_features)
        self.dW = grad_output.dot(self.x.transpose())
        
        if self.use_bias:
            # Gradient w.r.t. bias is just grad_output (shape (out_features, 1))
            self.db = grad_output
        
        # Gradient w.r.t. input: W^T @ grad_output
        # W^T is (in_features, out_features), grad_output is (out_features, 1)
        # Result is (in_features, 1)
        return self.W.transpose().dot(grad_output)
    
    def get_params(self):
        """Return parameters and their gradients"""
        params = [{'param': self.W, 'grad': self.dW, 'name': 'W'}]
        if self.use_bias:
            params.append({'param': self.b, 'grad': self.db, 'name': 'b'})
        return params
    
    def set_params(self, params):
        """Set updated parameters from optimizer"""
        for p in params:
            if p['name'] == 'W':
                self.W = p['param']
            elif p['name'] == 'b':
                self.b = p['param']


# Keep Linear as alias for backward compatibility
Linear = Dense


class Sequential(Layer):
    """Sequential container for layers"""
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def get_params(self):
        """Collect parameters from all layers"""
        params = []
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for p in layer_params:
                p['layer_idx'] = i # Tag with layer index
                params.append(p)
        return params
    
    def set_params(self, params):
        """Distribute updated parameters to layers"""
        layer_params = {}
        for p in params:
            idx = p['layer_idx']
            if idx not in layer_params:
                layer_params[idx] = []
            layer_params[idx].append(p)
        
        for idx, lp in layer_params.items():
            self.layers[idx].set_params(lp)

