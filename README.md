# 🧠 Neural Network Framework

A lightweight neural network framework built **from scratch in Python** without external libraries.  
It provides a custom `Matrix` implementation, neural network layers, loss functions, metrics, and a simple training loop — perfect for learning the internals of deep learning.

---

## ✨ Features

- **Matrix Class**
  - 2D matrix with operator overloading (`+`, `-`, `@`, etc.)
  - Utilities for initialization and manipulation

- **Layers**
  - Linear (Fully Connected)
  - Activation: ReLU, Sigmoid, Tanh
  - `Sequential` container for stacking layers

- **Loss**
  - Mean Squared Error (`MSELoss`)

- **Training Utilities**
  - `Trainer` class for forward/backward propagation and parameter updates
  - `Model` wrapper for a simple API

- **Metrics**
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Akay-74/Neural-Network-Framework.git
cd Neural-Network-Framework