"""
Digits Dataset Comparison: NNF vs TensorFlow (Fair Competition)
FIXED VERSION - Matching Iris test structure with all visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import time
import random
import seaborn as sns
from matplotlib.colors import ListedColormap

# Import refactored NNF framework
from NNF import (Tensor, Dense, ReLU, Softmax, Model, Trainer, 
                 CrossEntropyLoss, Adam, Accuracy)

plt.style.use('seaborn-v0_8-darkgrid')

def prepare_digits_data():
    """Load and preprocess Digits dataset"""
    print("Loading Digits dataset...")
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode labels
    n_classes = len(np.unique(y))
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]
    
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Classes: {n_classes}")
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            y_train_onehot, y_test_onehot, scaler)

def convert_to_nnf_format(X, y):
    """Convert numpy arrays to NNF Tensor format"""
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    return X_tensors, y_tensors

def create_nnf_model():
    """Create neural network using NNF framework"""
    print("\nCreating NNF model with Adam optimizer...")
    
    model = Model([
        Dense(64, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ])
    
    return model

def create_tensorflow_model():
    """Create IDENTICAL neural network using TensorFlow"""
    print("\nCreating TensorFlow model with Adam optimizer...")
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(64,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_nnf_model(model, X_train, y_train, epochs=100):
    """Train NNF model with Adam optimizer"""
    print(f"\nTraining NNF model for {epochs} epochs...")
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    
    # Track losses for plotting
    losses = []
    
    start_time = time.time()
    
    # Custom training loop to track losses
    n_samples = len(X_train)
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y_true in zip(X_train, y_train):
            y_pred = model.forward(x)
            loss = loss_fn.forward(y_pred, y_true)
            total_loss += loss
            
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)
            
            params = model.get_params()
            updated_params = optimizer.step(params)
            model.set_params(updated_params)
        
        avg_loss = total_loss / n_samples
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"NNF training completed in {training_time:.2f} seconds")
    
    return training_time, losses

def train_tensorflow_model(model, X_train, y_train, epochs=100):
    """Train TensorFlow model"""
    print(f"\nTraining TensorFlow model for {epochs} epochs...")
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=1,  # Same as NNF (processes one sample at a time)
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Print periodic updates
    for i in range(0, epochs, 10):
        if i < len(history.history['loss']):
            print(f"Epoch {i+1}/{epochs} | Loss: {history.history['loss'][i]:.6f}")
    
    print(f"TensorFlow training completed in {training_time:.2f} seconds")
    return history, training_time

def evaluate_nnf_model(model, X_test, y_test):
    """Evaluate NNF model"""
    print("\nEvaluating NNF model...")
    
    predictions = [model.forward(x) for x in X_test]
    
    pred_classes = [np.argmax([val[0] for val in pred.data]) for pred in predictions]
    true_classes = [np.argmax([val[0] for val in true.data]) for true in y_test]
    
    accuracy = accuracy_score(true_classes, pred_classes)
    return accuracy, pred_classes, true_classes

def evaluate_tensorflow_model(model, X_test, y_test_onehot, y_test):
    """Evaluate TensorFlow model"""
    print("\nEvaluating TensorFlow model...")
    
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, pred_classes)
    
    return accuracy, pred_classes

def plot_training_curves(nnf_losses, tf_losses):
    """Plot smooth training loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(nnf_losses) + 1)
    
    ax.plot(epochs, nnf_losses, label='NNF (Adam)', color='#2E86AB', linewidth=2.5, alpha=0.8)
    ax.plot(epochs, tf_losses, label='TensorFlow (Adam)', color='#A23B72', linewidth=2.5, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss Comparison (Digits Dataset)', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('digits_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_decision_boundaries(X_test, y_test, nnf_model, tf_model, scaler):
    """Plot beautiful PCA decision boundaries"""
    print("\nCreating PCA decision boundary visualization...")
    
    # Get full dataset for better PCA
    digits = load_digits()
    X_full = scaler.transform(digits.data)
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)
    X_test_pca = pca.transform(X_test)
    
    # Create mesh
    h = 0.5  # step size (larger for digits as feature space is bigger)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Inverse transform to original space
    mesh_points_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points_pca)
    
    # Predict with both models
    tf_preds = tf_model.predict(mesh_points_original, verbose=0)
    tf_classes = np.argmax(tf_preds, axis=1).reshape(xx.shape)
    
    print("Computing NNF predictions for decision boundary...")
    nnf_classes = []
    for i, pt in enumerate(mesh_points_original):
        if i % 500 == 0:
            print(f"  Processing point {i}/{len(mesh_points_original)}")
        x_tensor = Tensor([[float(v)] for v in pt])
        pred = nnf_model.forward(x_tensor)
        pred_class = np.argmax([p[0] for p in pred.data])
        nnf_classes.append(pred_class)
    nnf_classes = np.array(nnf_classes).reshape(xx.shape)
    
    # Color maps for 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # NNF Decision Boundary
    ax1 = axes[0]
    contour1 = ax1.contourf(xx, yy, nnf_classes, alpha=0.3, levels=np.arange(-0.5, 10, 1), 
                            cmap=plt.cm.tab10)
    for class_idx in range(10):
        mask = y_test == class_idx
        if mask.sum() > 0:
            ax1.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                       c=[colors[class_idx]], label=f'Digit {class_idx}', 
                       s=50, edgecolors='black', linewidth=1, alpha=0.8)
    ax1.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
    ax1.set_title('NNF Decision Boundary', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=8, frameon=True, shadow=True, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # TensorFlow Decision Boundary
    ax2 = axes[1]
    contour2 = ax2.contourf(xx, yy, tf_classes, alpha=0.3, levels=np.arange(-0.5, 10, 1),
                            cmap=plt.cm.tab10)
    for class_idx in range(10):
        mask = y_test == class_idx
        if mask.sum() > 0:
            ax2.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                       c=[colors[class_idx]], label=f'Digit {class_idx}', 
                       s=50, edgecolors='black', linewidth=1, alpha=0.8)
    ax2.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
    ax2.set_title('TensorFlow Decision Boundary', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=8, frameon=True, shadow=True, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('digits_decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(nnf_true, nnf_pred, tf_true, tf_pred):
    """Plot confusion matrices side by side"""
    class_names = [str(i) for i in range(10)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # NNF Confusion Matrix
    cm_nnf = confusion_matrix(nnf_true, nnf_pred)
    sns.heatmap(cm_nnf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('NNF Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # TensorFlow Confusion Matrix
    cm_tf = confusion_matrix(tf_true, tf_pred)
    sns.heatmap(cm_tf, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('TensorFlow Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('digits_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(nnf_acc, tf_acc, nnf_time, tf_time):
    """Beautiful metrics comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['NNF\n(Adam)', 'TensorFlow\n(Adam)']
    colors = ['#2E86AB', '#A23B72']
    
    # Accuracy
    bars1 = axes[0].bar(models, [nnf_acc, tf_acc], color=colors, 
                        width=0.6, edgecolor='black', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_title('Test Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    
    for bar, acc in zip(bars1, [nnf_acc, tf_acc]):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.4f}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    
    # Training Time
    bars2 = axes[1].bar(models, [nnf_time, tf_time], color=colors, 
                        width=0.6, edgecolor='black', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('Training Time (seconds)', fontsize=13, fontweight='bold')
    axes[1].set_title('Training Time Comparison', fontsize=15, fontweight='bold', pad=20)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    
    for bar, t in zip(bars2, [nnf_time, tf_time]):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{t:.2f}s', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('digits_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sample_predictions(X_test, y_test, nnf_pred, tf_pred):
    """Plot sample digits with predictions"""
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    
    for i in range(10):
        # Reshape from 64 features to 8x8 image
        img = X_test[i].reshape(8, 8)
        
        # NNF predictions
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        color = 'green' if nnf_pred[i] == y_test[i] else 'red'
        axes[0, i].set_title(f'P:{nnf_pred[i]}\nT:{y_test[i]}', 
                            fontsize=9, color=color, fontweight='bold')
        
        # TensorFlow predictions
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        color = 'green' if tf_pred[i] == y_test[i] else 'red'
        axes[1, i].set_title(f'P:{tf_pred[i]}\nT:{y_test[i]}', 
                            fontsize=9, color=color, fontweight='bold')
    
    axes[0, 0].set_ylabel('NNF', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('TensorFlow', fontsize=12, fontweight='bold')
    
    plt.suptitle('Sample Predictions (P=Predicted, T=True)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('digits_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=" * 70)
    print("DIGITS DATASET: FAIR COMPETITION")
    print("NNF vs TensorFlow (Same Architecture, Same Optimizer)")
    print("=" * 70)
    
    # Prepare data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, scaler = prepare_digits_data()
    
    # Convert to NNF format
    X_train_nnf, y_train_nnf = convert_to_nnf_format(X_train, y_train_onehot)
    X_test_nnf, y_test_nnf = convert_to_nnf_format(X_test, y_test_onehot)
    
    # Create identical models
    nnf_model = create_nnf_model()
    tf_model = create_tensorflow_model()
    
    epochs = 100
    print(f"\nTraining Parameters:")
    print(f"  Architecture: [64 -> 128 -> 64 -> 10]")
    print(f"  Optimizer: Adam (lr=0.01, beta1=0.9, beta2=0.999)")
    print(f"  Loss: Cross-Entropy")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: 1 (online learning)")
    
    # Train models
    nnf_time, nnf_losses = train_nnf_model(nnf_model, X_train_nnf, y_train_nnf, epochs)
    tf_history, tf_time = train_tensorflow_model(tf_model, X_train, y_train_onehot, epochs)
    
    # Evaluate models
    nnf_acc, nnf_pred, nnf_true = evaluate_nnf_model(nnf_model, X_test_nnf, y_test_nnf)
    tf_acc, tf_pred = evaluate_tensorflow_model(tf_model, X_test, y_test_onehot, y_test)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"NNF:")
    print(f"  Accuracy: {nnf_acc:.4f} ({nnf_acc*100:.2f}%)")
    print(f"  Training Time: {nnf_time:.2f}s")
    
    print(f"\nTensorFlow:")
    print(f"  Accuracy: {tf_acc:.4f} ({tf_acc*100:.2f}%)")
    print(f"  Training Time: {tf_time:.2f}s")
    
    print(f"\nDifference:")
    print(f"  Accuracy: {(nnf_acc - tf_acc)*100:+.2f}% (NNF - TF)")
    if nnf_time > tf_time:
        print(f"  Speed: {(nnf_time/tf_time):.2f}x slower (TF is faster)")
    else:
        print(f"  Speed: {(tf_time/nnf_time):.2f}x faster (NNF is faster)")
    
    print(f"\nNNF Classification Report:")
    print(classification_report(nnf_true, nnf_pred, 
                              target_names=[str(i) for i in range(10)]))
    
    print(f"\nTensorFlow Classification Report:")
    print(classification_report(y_test, tf_pred, 
                              target_names=[str(i) for i in range(10)]))
    
    # Create visualizations
    plot_training_curves(nnf_losses, tf_history.history['loss'])
    plot_pca_decision_boundaries(X_test, y_test, nnf_model, tf_model, scaler)
    plot_confusion_matrices(nnf_true, nnf_pred, y_test, tf_pred)
    plot_metrics_comparison(nnf_acc, tf_acc, nnf_time, tf_time)
    plot_sample_predictions(X_test, y_test, nnf_pred, tf_pred)
    
    print("\n" + "=" * 70)
    print("Comparison completed! Graphs saved:")
    print("  - digits_training_curves.png")
    print("  - digits_decision_boundaries.png")
    print("  - digits_confusion_matrices.png")
    print("  - digits_metrics_comparison.png")
    print("  - digits_sample_predictions.png")
    print("=" * 70)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()