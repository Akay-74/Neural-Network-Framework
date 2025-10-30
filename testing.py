import random
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import the NNF components needed for a manual loop
from NNF import (Tensor, Model, Dense, ReLU,
                 MSELoss, Adam)

def convert_to_nnf_format(X, y):
    """Converts numpy arrays to list of NNF column-vector Tensors"""
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    Y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    return X_tensors, Y_tensors

def main():
    print("--- Loading Diabetes Dataset ---")
    X_np, y_np = load_diabetes(return_X_y=True)
    
    # Reshape y to be (n_samples, 1) for scaling
    y_np = y_np.reshape(-1, 1)

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )

    # Scale all data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_np)
    X_test_scaled = scaler_X.transform(X_test_np)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_np)
    y_test_scaled = scaler_y.transform(y_test_np)
    
    # Convert scaled data to NNF format
    X_train_nnf, Y_train_nnf = convert_to_nnf_format(X_train_scaled, y_train_scaled)
    # X_test_nnf, Y_test_nnf = convert_to_nnf_format(X_test_scaled, y_test_scaled) # Not needed for just training

    n_features = X_train_np.shape[1]
    
    # --- Define Model and Training Components ---
    model = Model([
        Dense(n_features, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 1)
    ])

    loss_fn = MSELoss()
    optimizer = Adam(lr=0.001) # Using a smaller LR for more stable SGD
    epochs = 50 # Fewer epochs needed to see the pattern

    print("--- Starting Manual Training (to get squiggly curve) ---")
    
    # This list will store the loss for EVERY SINGLE SAMPLE
    per_sample_loss_history = []
    
    n_samples = len(X_train_nnf)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # Iterate sample by sample (Stochastic Gradient Descent)
        for x, y_true in zip(X_train_nnf, Y_train_nnf):
            
            # 1. Forward pass
            y_pred = model.forward(x)
            
            # 2. Compute loss
            loss = loss_fn.forward(y_pred, y_true)
            per_sample_loss_history.append(loss) # Store the squiggly data
            epoch_loss += loss
            
            # 3. Backward pass
            grad_loss = loss_fn.backward()
            model.backward(grad_loss)
            
            # 4. Optimizer step
            params = model.get_params()
            updated_params = optimizer.step(params)
            model.set_params(updated_params)
        
        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / n_samples
        print(f"Epoch {epoch}/{epochs} | Average Loss: {avg_epoch_loss:.6f}")


    # --- Plotting Results ---
    print("--- Plotting Per-Sample Loss Curve ---")
    
    plt.figure(figsize=(10, 6))
    # The x-axis is just the total number of samples processed
    plt.plot(range(len(per_sample_loss_history)), per_sample_loss_history, 
             label='Per-Sample Loss', color='blue', alpha=0.7, linewidth=1)
    
    plt.xlabel("Training Step (Sample by Sample)", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("NNF Model: 'Squiggly' Per-Sample Training Loss (Diabetes Dataset)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, max(per_sample_loss_history[n_samples:]) * 0.5) # Zoom in past the initial drop
    plt.show()

if __name__ == "__main__":
    random.seed(42) 
    np.random.seed(42)
    main()
