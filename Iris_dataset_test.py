"""
Iris Dataset Comparison: NNF vs TensorFlow (Simple Version)
This script provides a direct, simple comparison, focusing on
training time, final accuracy, continuous loss & accuracy graphs,
and a plot of actual vs. predicted classes.
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
# Make sure NNF.py is in the same directory
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
    # Also return original X_test for the new plot, and feature names
    return (X_train_scaled, X_test_scaled, y_train, y_test,
            y_train_onehot, y_test_onehot, iris.feature_names, X_test)

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
        
        # Shuffle data each epoch
        indices = list(range(n_samples))
        random.shuffle(indices)
        X_train_shuffled = [X_train[i] for i in indices]
        y_train_shuffled = [y_train[i] for i in indices]
        
        for x, y_true in zip(X_train_shuffled, y_train_shuffled):
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
        verbose=0,     # Suppress TF's own logger
        shuffle=True
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
    
    epochs_range = np.arange(1, len(nnf_losses) + 1)
    
    ax.plot(epochs_range, nnf_losses, label='NNF (Adam)', color='#2E86AB', linewidth=2)
    ax.plot(epochs_range, tf_losses, label='TensorFlow (Adam)', color='#A23B72', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison (Iris Dataset)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0) # Loss can't be negative
    
    plt.tight_layout()
    plt.savefig('iris_simple_training_curves.png', dpi=200)
    plt.show()
    print("Graph saved as 'iris_simple_training_curves.png'")

def plot_accuracy_curves(nnf_accuracies, tf_accuracies):
    """Plot the continuous test accuracy curves."""
    print("Generating training accuracy graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_range = np.arange(1, len(nnf_accuracies) + 1)
    
    ax.plot(epochs_range, nnf_accuracies, label='NNF (Adam)', color='#2E86AB', linewidth=2)
    ax.plot(epochs_range, tf_accuracies, label='TensorFlow (Adam)', color='#A23B72', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Comparison (Iris Dataset)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05) # Accuracy is between 0 and 1
    
    plt.tight_layout()
    plt.savefig('iris_simple_accuracy_curves.png', dpi=200)
    plt.show()
    print("Graph saved as 'iris_simple_accuracy_curves.png'")

def plot_prediction_comparison(X_test_original, y_test, nnf_preds, tf_preds, feature_names, feature_index=0):
    """
    Plots Actual vs. Predicted classes against a chosen input feature.
    X_test_original is the *unscaled* test data for easier interpretation.
    """
    print("\nGenerating prediction comparison graph...")
    
    feature_name = feature_names[feature_index]
    X_feature = X_test_original[:, feature_index]
    
    # Sort all arrays based on the chosen feature for a clean line plot
    sort_indices = np.argsort(X_feature)
    X_sorted = X_feature[sort_indices]
    y_actual_sorted = y_test[sort_indices]
    nnf_preds_sorted = nnf_preds[sort_indices]
    tf_preds_sorted = tf_preds[sort_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- NNF Plot ---
    ax1.plot(X_sorted, y_actual_sorted, 'o-', color='blue', label='Actual Class', markersize=5, linewidth=1, alpha=0.7)
    ax1.plot(X_sorted, nnf_preds_sorted, 'x--', color='red', label='NNF Predicted Class', markersize=5, linewidth=1, alpha=0.7)
    ax1.set_title('NNF: Actual vs. Predicted Class', fontsize=13)
    ax1.set_ylabel('Class (0, 1, 2)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks([0, 1, 2]) # Ensure ticks are only on class labels

    # --- TensorFlow Plot ---
    ax2.plot(X_sorted, y_actual_sorted, 'o-', color='blue', label='Actual Class', markersize=5, linewidth=1, alpha=0.7)
    ax2.plot(X_sorted, tf_preds_sorted, 'x--', color='red', label='TF Predicted Class', markersize=5, linewidth=1, alpha=0.7)
    ax2.set_title('TensorFlow: Actual vs. Predicted Class', fontsize=13)
    ax2.set_xlabel(f'Input Feature: {feature_name} (sorted)', fontsize=12)
    ax2.set_ylabel('Class (0, 1, 2)', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks([0, 1, 2])

    fig.suptitle(f'Prediction Comparison vs. {feature_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
    plt.savefig('iris_simple_prediction_comparison.png', dpi=200)
    plt.show()
    print("Graph saved as 'iris_simple_prediction_comparison.png'")


def main():
    print("=" * 70)
    print("Simple NNF vs TensorFlow Comparison (Iris Dataset)")
    print("=" * 70)
    
    # 1. Prepare data
    (X_train_scaled, X_test_scaled, y_train, y_test,
     y_train_onehot, y_test_onehot,
     feature_names, X_test_original) = prepare_iris_data()
    
    # 2. Convert data for NNF
    X_train_nnf, y_train_nnf = convert_to_nnf_format(X_train_scaled, y_train_onehot)
    X_test_nnf, _ = convert_to_nnf_format(X_test_scaled, y_test_onehot)
    
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
        tf_model, X_train_scaled, y_train_onehot, X_test_scaled, y_test_onehot, epochs
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
    
    # 8. Get final predictions for the new plot
    nnf_predictions = [nnf_model.forward(x) for x in X_test_nnf]
    nnf_pred_classes = np.array([np.argmax([val[0] for val in pred.data]) for pred in nnf_predictions])
    
    tf_pred_probs = tf_model.predict(X_test_scaled, verbose=0)
    tf_pred_classes = np.argmax(tf_pred_probs, axis=1)
    
    # 9. Plot the new actual vs. predicted graph
    # We use X_test_original (unscaled) for a more interpretable X-axis
    plot_prediction_comparison(
        X_test_original, y_test, nnf_pred_classes, tf_pred_classes, feature_names, feature_index=0
    )
    
    print("\n" + "=" * 70)
    print("Simple comparison completed.")
    print("=" * 70)

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Run the main comparison
    main()
