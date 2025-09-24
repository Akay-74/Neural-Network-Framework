"""
Digits Dataset Comparison: NNF vs TensorFlow
Compare performance of custom NNF framework against TensorFlow on Digits classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
import time

# Import your NNF framework
from NNF import Tensor, Linear, ReLU, Sigmoid, Model, Trainer, MSELoss

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
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_train_onehot, y_test_onehot

def convert_to_nnf_format(X, y):
    """Convert numpy arrays to NNF Tensor format"""
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    return X_tensors, y_tensors

def create_nnf_model(input_dim, output_dim):
    """Create neural network using NNF framework"""
    print("\nCreating NNF model...")
    
    model = Model([
        Linear(input_dim, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, output_dim),
        Sigmoid()  # Using sigmoid to match your original setup
    ])
    
    return model

def create_tensorflow_model(input_dim, output_dim):
    """Create equivalent neural network using TensorFlow"""
    print("\nCreating TensorFlow model...")
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss='mse',  # Using MSE to match NNF
        metrics=['accuracy']
    )
    
    return model

def train_nnf_model(model, X_train, y_train, epochs=100, lr=0.01):
    print(f"\nTraining NNF model for {epochs} epochs...")
    loss_fn = MSELoss()
    trainer = Trainer(model, loss_fn, lr=lr, epochs=epochs)
    
    start_time = time.time()
    trainer.fit(X_train, y_train, verbose=True)
    training_time = time.time() - start_time
    
    print(f"NNF training completed in {training_time:.2f} seconds")
    return training_time

def train_tensorflow_model(model, X_train, y_train, epochs=100):
    print(f"\nTraining TensorFlow model for {epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=1,  # batch_size=1 to mimic NNF sequential updates
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"TensorFlow training completed in {training_time:.2f} seconds")
    return history, training_time

def evaluate_nnf_model(model, X_test, y_test):
    print("\nEvaluating NNF model...")
    
    predictions = [model.forward(x) for x in X_test]
    
    pred_classes, true_classes = [], []
    for pred, true in zip(predictions, y_test):
        pred_class = np.argmax([val[0] for val in pred.data])
        true_class = np.argmax([val[0] for val in true.data])
        pred_classes.append(pred_class)
        true_classes.append(true_class)
    
    accuracy = sum(p == t for p, t in zip(pred_classes, true_classes)) / len(pred_classes)
    return accuracy, pred_classes, true_classes

def evaluate_tensorflow_model(model, X_test, y_test_onehot, y_test):
    print("\nEvaluating TensorFlow model...")
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, pred_classes)
    return accuracy, pred_classes

def plot_comparison(nnf_accuracy, tf_accuracy, nnf_time, tf_time):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    
    models = ['NNF', 'TensorFlow']
    colors = ['skyblue', 'lightcoral']
    
    # Accuracy
    bars1 = ax1.bar(models, [nnf_accuracy, tf_accuracy], color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, [nnf_accuracy, tf_accuracy]):
        ax1.text(bar.get_x()+bar.get_width()/2, acc+0.01, f'{acc:.3f}', ha='center')
    
    # Training time
    bars2 = ax2.bar(models, [nnf_time, tf_time], color=colors)
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time Comparison')
    for bar, t in zip(bars2, [nnf_time, tf_time]):
        ax2.text(bar.get_x()+bar.get_width()/2, t+0.01, f'{t:.2f}s', ha='center')
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("DIGITS DATASET COMPARISON: NNF vs TensorFlow")
    print("="*60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = prepare_digits_data()
    
    # Convert to NNF format
    X_train_nnf, y_train_nnf = convert_to_nnf_format(X_train, y_train_onehot)
    X_test_nnf, y_test_nnf = convert_to_nnf_format(X_test, y_test_onehot)
    
    input_dim = X_train.shape[1]
    output_dim = y_train_onehot.shape[1]
    
    # Create models
    nnf_model = create_nnf_model(input_dim, output_dim)
    tf_model = create_tensorflow_model(input_dim, output_dim)
    
    epochs = 100
    learning_rate = 0.01
    
    # Train models
    nnf_time = train_nnf_model(nnf_model, X_train_nnf, y_train_nnf, epochs, learning_rate)
    tf_history, tf_time = train_tensorflow_model(tf_model, X_train, y_train_onehot, epochs)
    
    # Evaluate models
    nnf_acc, nnf_pred, nnf_true = evaluate_nnf_model(nnf_model, X_test_nnf, y_test_nnf)
    tf_acc, tf_pred = evaluate_tensorflow_model(tf_model, X_test, y_test_onehot, y_test)
    
    # Print results
    print("\nFINAL RESULTS")
    print(f"NNF: Accuracy={nnf_acc:.4f}, Training Time={nnf_time:.2f}s")
    print(f"TensorFlow: Accuracy={tf_acc:.4f}, Training Time={tf_time:.2f}s")
    
    print(f"\nNNF Classification Report:")
    print(classification_report(nnf_true, nnf_pred))
    print(f"\nTensorFlow Classification Report:")
    print(classification_report(y_test, tf_pred))
    
    # Plot comparison
    plot_comparison(nnf_acc, tf_acc, nnf_time, tf_time)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
