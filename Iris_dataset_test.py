"""
Iris Dataset Comparison: Neural Network Framework (NNF) vs TensorFlow
Compare performance of custom NNF framework against TensorFlow on Iris classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
import time
import random
# Import your NNF framework
from NNF import Tensor, Linear, ReLU, Sigmoid, Model, Trainer, MSELoss, Accuracy

def prepare_iris_data():
    """Load and preprocess Iris dataset"""
    print("Loading Iris dataset...")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert labels to one-hot for multi-class classification
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]
    
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            y_train_onehot, y_test_onehot)

def convert_to_nnf_format(X, y):
    """Convert numpy arrays to NNF Tensor format"""
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    return X_tensors, y_tensors

def create_nnf_model():
    """Create neural network using NNF framework"""
    print("\nCreating Neural Network Framework (NNF) model...")
    
    # Simple 3-layer network: 4 -> 8 -> 4 -> 3
    model = Model([
        Linear(4, 8),
        ReLU(),
        Linear(8, 4),
        ReLU(),
        Linear(4, 3),
        Sigmoid()
    ])
    
    return model

def create_tensorflow_model():
    """Create equivalent neural network using TensorFlow"""
    print("\nCreating TensorFlow model...")
    
    model = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(3, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )
    
    return model

def train_nnf_model(model, X_train, y_train, epochs=100, lr=0.01):
    """Train NNF model and track metrics"""
    print(f"\nTraining Neural Network Framework (NNF) model for {epochs} epochs...")
    
    loss_fn = MSELoss()
    trainer = Trainer(model, loss_fn, lr=lr, epochs=epochs)
    
    start_time = time.time()
    trainer.fit(X_train, y_train, verbose=True)
    training_time = time.time() - start_time
    
    print(f"Neural Network Framework (NNF) training completed in {training_time:.2f} seconds")
    return training_time

def train_tensorflow_model(model, X_train, y_train, epochs=100):
    """Train TensorFlow model and track metrics"""
    print(f"\nTraining TensorFlow model for {epochs} epochs...")
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_split=0.2
    )
    training_time = time.time() - start_time
    
    print(f"TensorFlow training completed in {training_time:.2f} seconds")
    return history, training_time

def evaluate_nnf_model(model, X_test, y_test):
    """Evaluate NNF model performance"""
    print("\nEvaluating Neural Network Framework (NNF) model...")
    
    # Get predictions
    predictions = []
    for x in X_test:
        pred = model.forward(x)
        predictions.append(pred)
    
    # Convert predictions to class labels
    pred_classes = []
    true_classes = []
    
    for pred, true in zip(predictions, y_test):
        # Find class with highest probability
        pred_class = 0
        max_prob = pred.data[0][0]
        for i in range(len(pred.data)):
            if pred.data[i][0] > max_prob:
                max_prob = pred.data[i][0]
                pred_class = i
        pred_classes.append(pred_class)
        
        # Find true class
        true_class = 0
        for i in range(len(true.data)):
            if true.data[i][0] == 1.0:
                true_class = i
                break
        true_classes.append(true_class)
    
    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(pred_classes, true_classes)) / len(pred_classes)
    
    return accuracy, pred_classes, true_classes, predictions

def evaluate_tensorflow_model(model, X_test, y_test_onehot, y_test):
    """Evaluate TensorFlow model performance"""
    print("\nEvaluating TensorFlow model...")
    
    # Get predictions
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred_classes)
    
    return accuracy, pred_classes, predictions

def plot_comparison(nnf_accuracy, tf_accuracy, nnf_time, tf_time):
    """Create comparison plots for accuracy and training time"""
    print("\nCreating comparison plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    models = ['Neural Network\nFramework (NNF)', 'TensorFlow']
    accuracies = [nnf_accuracy, tf_accuracy]
    colors = ['#4A90E2', '#E74C3C']
    
    bars1 = ax1.bar(models, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Training time comparison
    times = [nnf_time, tf_time]
    bars2 = ax2.bar(models, times, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.02,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_comparison(X_test, y_test, nnf_pred, tf_pred, nnf_pred_probs, tf_pred_probs):
    """Create visualization comparing actual vs predicted classes for both models"""
    print("\nCreating predictions comparison visualization...")
    
    # Use first two features for 2D visualization (Sepal length and Sepal width)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    # Plot 1: Actual Data Points
    ax1 = axes[0]
    for class_idx in range(3):
        mask = y_test == class_idx
        ax1.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=colors[class_idx], marker=markers[class_idx],
                   label=class_names[class_idx], s=100, 
                   edgecolors='black', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Sepal Length (standardized)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sepal Width (standardized)', fontsize=11, fontweight='bold')
    ax1.set_title('Actual Data Points', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Neural Network Framework (NNF) Predictions
    ax2 = axes[1]
    for class_idx in range(3):
        mask = np.array(nnf_pred) == class_idx
        # Mark correct and incorrect predictions
        correct_mask = mask & (np.array(nnf_pred) == y_test)
        incorrect_mask = mask & (np.array(nnf_pred) != y_test)
        
        if np.any(correct_mask):
            ax2.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1],
                       c=colors[class_idx], marker=markers[class_idx],
                       label=f'{class_names[class_idx]} (Correct)', s=100,
                       edgecolors='black', linewidth=1.5, alpha=0.7)
        if np.any(incorrect_mask):
            ax2.scatter(X_test[incorrect_mask, 0], X_test[incorrect_mask, 1],
                       c=colors[class_idx], marker='x',
                       label=f'{class_names[class_idx]} (Wrong)', s=150,
                       edgecolors='red', linewidth=3)
    
    ax2.set_xlabel('Sepal Length (standardized)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sepal Width (standardized)', fontsize=11, fontweight='bold')
    ax2.set_title('Neural Network Framework (NNF)\nPredictions', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: TensorFlow Predictions
    ax3 = axes[2]
    for class_idx in range(3):
        mask = tf_pred == class_idx
        # Mark correct and incorrect predictions
        correct_mask = mask & (tf_pred == y_test)
        incorrect_mask = mask & (tf_pred != y_test)
        
        if np.any(correct_mask):
            ax3.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1],
                       c=colors[class_idx], marker=markers[class_idx],
                       label=f'{class_names[class_idx]} (Correct)', s=100,
                       edgecolors='black', linewidth=1.5, alpha=0.7)
        if np.any(incorrect_mask):
            ax3.scatter(X_test[incorrect_mask, 0], X_test[incorrect_mask, 1],
                       c=colors[class_idx], marker='x',
                       label=f'{class_names[class_idx]} (Wrong)', s=150,
                       edgecolors='red', linewidth=3)
    
    ax3.set_xlabel('Sepal Length (standardized)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sepal Width (standardized)', fontsize=11, fontweight='bold')
    ax3.set_title('TensorFlow Predictions', fontsize=13, fontweight='bold', pad=15)
    ax3.legend(loc='best', frameon=True, shadow=True, fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main comparison function"""
    print("=" * 70)
    print("IRIS DATASET COMPARISON")
    print("Neural Network Framework (NNF) vs TensorFlow")
    print("=" * 70)
    
    # Prepare data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = prepare_iris_data()
    
    # Convert to NNF format
    X_train_nnf, y_train_nnf = convert_to_nnf_format(X_train, y_train_onehot)
    X_test_nnf, y_test_nnf = convert_to_nnf_format(X_test, y_test_onehot)
    
    # Create models
    nnf_model = create_nnf_model()
    tf_model = create_tensorflow_model()
    
    # Training parameters
    epochs = 100
    learning_rate = 0.01
    
    print(f"\nTraining Parameters:")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    
    # Train NNF model
    nnf_training_time = train_nnf_model(nnf_model, X_train_nnf, y_train_nnf, 
                                       epochs=epochs, lr=learning_rate)
    
    # Train TensorFlow model
    tf_history, tf_training_time = train_tensorflow_model(tf_model, X_train, y_train_onehot, 
                                                         epochs=epochs)
    
    # Evaluate models
    nnf_accuracy, nnf_pred, nnf_true, nnf_pred_probs = evaluate_nnf_model(nnf_model, X_test_nnf, y_test_nnf)
    tf_accuracy, tf_pred, tf_pred_probs = evaluate_tensorflow_model(tf_model, X_test, y_test_onehot, y_test)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Neural Network Framework (NNF):")
    print(f"  - Accuracy: {nnf_accuracy:.4f}")
    print(f"  - Training Time: {nnf_training_time:.2f} seconds")
    
    print(f"\nTensorFlow:")
    print(f"  - Accuracy: {tf_accuracy:.4f}")
    print(f"  - Training Time: {tf_training_time:.2f} seconds")
    
    print(f"\nComparison:")
    accuracy_diff = nnf_accuracy - tf_accuracy
    time_diff = nnf_training_time - tf_training_time
    print(f"  - Accuracy Difference: {accuracy_diff:+.4f} (NNF - TensorFlow)")
    print(f"  - Time Difference: {time_diff:+.2f}s (NNF - TensorFlow)")
    
    # Detailed classification report for both models
    print(f"\nNeural Network Framework (NNF) Classification Report:")
    print(classification_report(nnf_true, nnf_pred, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    print(f"\nTensorFlow Classification Report:")
    print(classification_report(y_test, tf_pred, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # Create comparison plots
    plot_comparison(nnf_accuracy, tf_accuracy, nnf_training_time, tf_training_time)
    
    # Create predictions comparison visualization
    plot_predictions_comparison(X_test, y_test, nnf_pred, tf_pred, nnf_pred_probs, tf_pred_probs)
    
    print("\nComparison completed!")
    print("Graphs saved as 'model_comparison.png' and 'predictions_comparison.png'")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    main()