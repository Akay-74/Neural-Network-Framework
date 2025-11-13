# Neural Network Framework (NNF)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview
**Neural Network Framework (NNF)** is a modular deep learning library built from scratch in Python.  
It aims to help users understand the inner workings of neural networks — including layers, activations, optimizers, and training logic — without relying on high-level frameworks like TensorFlow or PyTorch.

---

## Features
- **Modular Architecture** – Plug-and-play components (layers, activations, losses, optimizers)
- **Custom Tensor Engine** – Lightweight implementation of tensor operations
- **Training Pipeline** – Built-in Trainer with epoch, batch, and loss tracking
- **Polymorphism Support** –  
  - Compile-time (via function overloading)  
  - Runtime (via abstract base classes)
- **Exception Handling** integrated across core modules
- **Metrics Tracking** – Accuracy, precision, and loss monitoring
- **Extensible Design** – Easy to add new layers, activations, or optimizers

---

## Project Structure
```
NeuralNetworkFramework/
│
├── core/
│ ├── model.py # Model class – forward pass, layer management
│ ├── layers.py # Layer definitions (Dense, Conv, etc.)
│ ├── tensor.py # Custom Tensor operations
│ ├── trainer.py # Training loop, backpropagation, and logging
│
├── components/
│ ├── activations.py # ReLU, Sigmoid, Softmax, etc.
│ ├── optimizers.py # SGD, Adam implementations
│ ├── losses.py # MSE, CrossEntropy
│ ├── metrics.py # Accuracy, Precision, Recall
│
├── utils/
│ ├── helpers.py # Utility functions
│ └── exceptions.py # Custom error handling
│
├── docs/
│ ├── UML_Diagram.mmd # Class diagram (Mermaid)
│ ├── UseCase.mmd # Use case diagram
│ └── README.md # Documentation
│
└── tests/
├── test_layers.py
├── test_model.py
└── test_trainer.py
```

---

## Current Progress (as of Nov 2025)

| Component | Description |
|------------|--------------|
| **Model** | Model creation, training pipeline, and forward propagation |
| **Layers** | Base and derived classes for neural layers |
| **Tensor** | Core data structure for mathematical operations |
| **Losses** | Implements MSE, CrossEntropy |
| **Trainer** | Manages training loop and gradient updates |
| **Metrics** | Computes accuracy, precision, recall |
| **Activation** | Implements ReLU, Sigmoid, Softmax |
| **Optimizer** | Implements SGD, Adam |
| **Utils** | Helper utilities and configuration management |

---

## UML & Design Progress
- **Class Diagram:** Complete  
- **Use Case Diagram:** Complete  
- **Sequence Diagram:** In progress  
- **Exception Handling:** Integrated  
- **Polymorphism:** Compile-time and runtime support

---

## Tech Stack
| Category | Tool |
|-----------|------|
| **Language** | Python 3.10+ |
| **Dependencies** | NumPy, Matplotlib (optional) |

---

## Getting Started

### Prerequisites
Make sure you have:
- **Python 3.10+**
- **pip** installed

Check versions:
```
python3 --version
pip3 --version
```

## Clone the Repository
```
git clone https://github.com/<your-username>/NeuralNetworkFramework.git
cd NeuralNetworkFramework
```

## Install Dependencies
```
pip install numpy matplotlib
```

## Example Usage
You can create a simple neural network using the framework like this:

```
from core.model import Model
from core.layers import Dense
from components.activations import ReLU, Softmax
from components.losses import CrossEntropy
from components.optimizers import SGD
from core.trainer import Trainer

# Define a simple neural network
model = Model([
    Dense(4, 8, activation=ReLU()),
    Dense(8, 3, activation=Softmax())
])

# Sample input (X) and output (y)
X = [[0.2, 0.3, 0.5, 0.7],
     [0.1, 0.8, 0.5, 0.9]]

y = [[1, 0, 0],
     [0, 1, 0]]

# Define loss and optimizer
loss = CrossEntropy()
optimizer = SGD(lr=0.01)

# Create trainer and train the model
trainer = Trainer(model, optimizer, loss)
trainer.train(X, y, epochs=50)
```

---

## Our Next Steps
- [ ] Implement `Sequential` model wrapper  
- [ ] Add visualization utilities (loss/accuracy plots)  
- [ ] Expand test coverage for all modules  
- [ ] Add sample notebooks demonstrating a working NN  

---

## Contributors
| Name | Major Contributions |
|------|----------------------|
| **Shreyash** | Model, Layers, UML Design |
| **Aayaan** | Activation, Optimizer, Utils |
| **Shlok** | Tensor, Losses |
| **Shubhankar** | Trainer, Metrics |

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Special thanks to all contributors and mentors who guided the design and structure of the framework.  
Developed as part of an academic AI/ML project to explore the fundamentals of deep learning.

---