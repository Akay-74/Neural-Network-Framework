"""
Iris Dataset Comparison: NNF vs TensorFlow
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
    print("\nCreating NNF model...")
    
    # Simple 3-layer network: 4 -> 8 -> 4 -> 3
    model = Model([
        Linear(4, 8),
        ReLU(),
        Linear(8, 4),
        ReLU(),
        Linear(4, 3),
        Sigmoid()  # Using sigmoid for output (could use softmax but not implemented)
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
        loss='mse',  # Using MSE to match NNF
        metrics=['accuracy']
    )
    
    return model

def train_nnf_model(model, X_train, y_train, epochs=100, lr=0.01):
    """Train NNF model and track metrics"""
    print(f"\nTraining NNF model for {epochs} epochs...")
    
    loss_fn = MSELoss()
    trainer = Trainer(model, loss_fn, lr=lr, epochs=epochs)
    
    start_time = time.time()
    trainer.fit(X_train, y_train, verbose=True)
    training_time = time.time() - start_time
    
    print(f"NNF training completed in {training_time:.2f} seconds")
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
    print("\nEvaluating NNF model...")
    
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
    
    return accuracy, pred_classes, true_classes

def evaluate_tensorflow_model(model, X_test, y_test_onehot, y_test):
    """Evaluate TensorFlow model performance"""
    print("\nEvaluating TensorFlow model...")
    
    # Get predictions
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred_classes)
    
    return accuracy, pred_classes

def plot_comparison(nnf_accuracy, tf_accuracy, nnf_time, tf_time):
    """Create comparison plots"""
    print("\nCreating comparison plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['NNF', 'TensorFlow']
    accuracies = [nnf_accuracy, tf_accuracy]
    colors = ['skyblue', 'lightcoral']
    
    bars1 = ax1.bar(models, accuracies, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    times = [nnf_time, tf_time]
    bars2 = ax2.bar(models, times, color=colors)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main comparison function"""
    print("=" * 60)
    print("IRIS DATASET COMPARISON: NNF vs TensorFlow")
    print("=" * 60)
    
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
    nnf_accuracy, nnf_pred, nnf_true = evaluate_nnf_model(nnf_model, X_test_nnf, y_test_nnf)
    tf_accuracy, tf_pred = evaluate_tensorflow_model(tf_model, X_test, y_test_onehot, y_test)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"NNF Framework:")
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
    print(f"\nNNF Classification Report:")
    print(classification_report(nnf_true, nnf_pred, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    print(f"\nTensorFlow Classification Report:")
    print(classification_report(y_test, tf_pred, 
                              target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    # Create comparison plots
    plot_comparison(nnf_accuracy, tf_accuracy, nnf_training_time, tf_training_time)
    
    print("\nComparison completed!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42) 

    main()  