"""
Iris Dataset Comparison: NNF vs TensorFlow (Simple Version)
This script provides a direct, simple comparison, focusing only on
training time, final accuracy, and continuous loss & accuracy graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import time
import random

# Import refactored NNF framework
from NNF import (Tensor, Dense, ReLU, Softmax, Model,
                 CrossEntropyLoss, Adam)

plt.style.use('seaborn-v0_8-darkgrid')

def prepare_iris_data():
    """Load, split, and scale the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode labels
    n_classes = len(np.unique(y))
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]
    
    print(f"Features: {X_train_scaled.shape[1]}, Classes: {n_classes}")
    # Return y_test (original labels) for easy accuracy calculation
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            y_train_onehot, y_test_onehot)

def convert_to_nnf_format(X, y):
    """Convert numpy arrays to NNF Tensor format (column vectors)."""
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    
    # Diagnostic print to confirm shape
    if X_tensors:
        print(f"First NNF input tensor shape: {X_tensors[0].shape}")
    return X_tensors, y_tensors

def create_nnf_model():
    """Create neural network using NNF framework."""
    print("Creating NNF model...")
    model = Model([
        Dense(4, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    ])
    return model

def create_tensorflow_model():
    """Create IDENTICAL neural network using TensorFlow."""
    print("Creating TensorFlow model...")
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'] # TF will track accuracy for us
    )
    return model

def evaluate_nnf_accuracy(model, X_test_nnf, y_test_original):
    """Helper function to get NNF accuracy during training."""
    predictions = [model.forward(x) for x in X_test_nnf]
    pred_classes = [np.argmax([val[0] for val in pred.data]) for pred in predictions]
    return accuracy_score(y_test_original, pred_classes)

def train_nnf_model(model, X_train, y_train, X_test, y_test_original, epochs=200):
    """Train NNF model and track continuous loss and accuracy."""
    print(f"\nTraining NNF model for {epochs} epochs...")
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.01)
    
    losses = []
    accuracies = [] # New: track continuous accuracy
    start_time = time.time()
    n_samples = len(X_train)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y_true in zip(X_train, y_train):
            # Forward pass
            y_pred = model.forward(x)
            loss = loss_fn.forward(y_pred, y_true)
            total_loss += loss
            
            # Backward pass
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)
            
            # Optimizer step
            params = model.get_params()
            updated_params = optimizer.step(params)
            model.set_params(updated_params)
        
        # --- Track metrics for this epoch ---
        avg_loss = total_loss / n_samples
        losses.append(avg_loss)
        
        # Evaluate on test set
        epoch_acc = evaluate_nnf_accuracy(model, X_test, y_test_original)
        accuracies.append(epoch_acc)
        # ---
        
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | Test Accuracy: {epoch_acc:.4f}")
    
    training_time = time.time() - start_time
    print(f"NNF training completed in {training_time:.2f} seconds")
    return training_time, losses, accuracies

def train_tensorflow_model(model, X_train, y_train, X_test, y_test_onehot, epochs=200):
    """Train TensorFlow model and track continuous loss and accuracy."""
    print(f"\nTraining TensorFlow model for {epochs} epochs...")
    start_time = time.time()
    
    # Use validation_data to track test accuracy continuously
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=1,  # Match NNF (online learning)
        validation_data=(X_test, y_test_onehot), # Track test metrics
        verbose=0      # Suppress TF's own logger
    )
    training_time = time.time() - start_time
    
    # Print our own log
    for epoch in [1, 20, 50, 100, 150, epochs]:
        idx = epoch - 1
        if idx < len(history.history['loss']):
            print(f"  Epoch {epoch}/{epochs} | Loss: {history.history['loss'][idx]:.6f} | Test Accuracy: {history.history['val_accuracy'][idx]:.4f}")

    print(f"TensorFlow training completed in {training_time:.2f} seconds")
    # Return training loss, validation accuracy, and time
    return history.history['loss'], history.history['val_accuracy'], training_time

def plot_training_curves(nnf_losses, tf_losses):
    """Plot the continuous training loss curves."""
    print("\nGenerating training loss graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(nnf_losses) + 1)
    
    ax.plot(epochs, nnf_losses, label='NNF (Adam)', color='#2E86AB', linewidth=2)
    ax.plot(epochs, tf_losses, label='TensorFlow (Adam)', color='#A23B72', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison (Iris Dataset)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_simple_training_curves.png', dpi=200)
    plt.show()
    print("Graph saved as 'iris_simple_training_curves.png'")

def plot_accuracy_curves(nnf_accuracies, tf_accuracies):
    """Plot the continuous test accuracy curves."""
    print("Generating training accuracy graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(nnf_accuracies) + 1)
    
    ax.plot(epochs, nnf_accuracies, label='NNF (Adam)', color='#2E86AB', linewidth=2)
    ax.plot(epochs, tf_accuracies, label='TensorFlow (Adam)', color='#A23B72', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Comparison (Iris Dataset)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_simple_accuracy_curves.png', dpi=200)
    plt.show()
    print("Graph saved as 'iris_simple_accuracy_curves.png'")

def main():
    print("=" * 70)
    print("Simple NNF vs TensorFlow Comparison (Iris Dataset)")
    print("=" * 70)
    
    # 1. Prepare data
    (X_train, X_test, y_train, y_test,
     y_train_onehot, y_test_onehot) = prepare_iris_data()
    
    # 2. Convert data for NNF
    X_train_nnf, y_train_nnf = convert_to_nnf_format(X_train, y_train_onehot)
    X_test_nnf, y_test_nnf = convert_to_nnf_format(X_test, y_test_onehot)
    
    # 3. Create identical models
    nnf_model = create_nnf_model()
    tf_model = create_tensorflow_model()
    
    epochs = 200
    
    # 4. Train models
    # Pass test data to training functions to track accuracy
    nnf_time, nnf_losses, nnf_accuracies = train_nnf_model(
        nnf_model, X_train_nnf, y_train_nnf, X_test_nnf, y_test, epochs
    )
    tf_losses, tf_accuracies, tf_time = train_tensorflow_model(
        tf_model, X_train, y_train_onehot, X_test, y_test_onehot, epochs
    )
    
    # 5. Get final accuracies
    nnf_acc = nnf_accuracies[-1]
    tf_acc = tf_accuracies[-1]
    
    # 6. Print simple results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"NNF Model:")
    print(f"  Final Accuracy: {nnf_acc*100:.2f}%")
    print(f"  Training Time: {nnf_time:.2f}s")
    
    print(f"\nTensorFlow Model:")
    print(f"  Final Accuracy: {tf_acc*100:.2f}%")
    print(f"  Training Time: {tf_time:.2f}s")
    
    # 7. Plot the continuous graphs
    plot_training_curves(nnf_losses, tf_losses)
    plot_accuracy_curves(nnf_accuracies, tf_accuracies) # New plot
    
    print("\n" + "=" * 70)
    print("Simple comparison completed.")
    print("=" * 70)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()

