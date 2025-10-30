"""
Optimizers for training neural networks (TensorFlow-style)
"""
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers"""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, params):
        """Update parameters based on gradients"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset optimizer state if needed"""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, lr=0.01):
        super().__init__(lr)
    
    def step(self, params):
        """Simple SGD update: param = param - lr * grad"""
        updated_params = []
        for p in params:
            if p['grad'] is not None:
                updated_param = p['param'] - p['grad'] * self.lr
                p['param'] = updated_param
            updated_params.append(p)
        return updated_params


class Momentum(Optimizer):
    """SGD with momentum optimizer"""
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, params):
        """Momentum update: v = momentum * v - lr * grad, param = param + v"""
        updated_params = []
        for p in params:
            if p['grad'] is not None:
                # Create unique key for this parameter
                # BUG FIX: Use a stable key (layer index, param name) instead of id()
                key = (p['layer_idx'], p['name'])
                
                # Initialize velocity if needed
                if key not in self.velocity:
                    self.velocity[key] = p['grad'] * 0  # Zero tensor with same shape
                
                # Update velocity and parameter
                self.velocity[key] = self.velocity[key] * self.momentum - p['grad'] * self.lr
                p['param'] = p['param'] + self.velocity[key]
            
            updated_params.append(p)
        return updated_params


class RMSprop(Optimizer):
    """RMSprop optimizer with adaptive learning rates"""
    def __init__(self, lr=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
    
    def step(self, params):
        """RMSprop update with squared gradient accumulation"""
        updated_params = []
        for p in params:
            if p['grad'] is not None:
                # BUG FIX: Use a stable key (layer index, param name) instead of id()
                key = (p['layer_idx'], p['name'])
                
                # Initialize cache if needed
                if key not in self.cache:
                    self.cache[key] = p['grad'] * p['grad'] * 0  # Zero tensor
                
                # Update cache (moving average of squared gradients)
                grad_squared = p['grad'] * p['grad']
                self.cache[key] = self.cache[key] * self.rho + grad_squared * (1 - self.rho)
                
                # Update parameter
                # param = param - lr * grad / (sqrt(cache) + epsilon)
                adjusted_grad = p['grad'] * self.lr
                denominator = self.cache[key].apply(lambda x: (x ** 0.5) + self.epsilon)
                update = adjusted_grad / denominator
                p['param'] = p['param'] - update
            
            updated_params.append(p)
        return updated_params


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def step(self, params):
        """Adam update with bias-corrected moment estimates"""
        self.t += 1
        updated_params = []
        
        for p in params:
            if p['grad'] is not None:
                # BUG FIX: Use a stable key (layer index, param name) instead of id()
                key = (p['layer_idx'], p['name'])
                
                # Initialize moments if needed
                if key not in self.m:
                    self.m[key] = p['grad'] * 0  # Zero tensor
                    self.v[key] = p['grad'] * 0  # Zero tensor
                
                # Update biased first moment estimate
                self.m[key] = self.m[key] * self.beta1 + p['grad'] * (1 - self.beta1)
                
                # Update biased second raw moment estimate
                grad_squared = p['grad'] * p['grad']
                self.v[key] = self.v[key] * self.beta2 + grad_squared * (1 - self.beta2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # Update parameter
                denominator = v_hat.apply(lambda x: (x ** 0.5) + self.epsilon)
                update = m_hat * self.lr / denominator
                p['param'] = p['param'] - update
            
            updated_params.append(p)
        return updated_params
    
    def zero_grad(self):
        """Reset time step (optional, usually not needed)"""
        # We don't reset m and v to maintain momentum across epochs
        pass