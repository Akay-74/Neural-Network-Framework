import random
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NNF import (Tensor, Model, Dense, ReLU,
                 MSELoss, Adam, Trainer)

def convert_to_nnf_format(X, y):
    X_tensors = [Tensor([[float(val)] for val in sample]) for sample in X]
    Y_tensors = [Tensor([[float(val)] for val in sample]) for sample in y]
    return X_tensors, Y_tensors

def main():
    X_np, y_np = load_diabetes(return_X_y=True)
    y_np = y_np.reshape(-1, 1)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_np)
    X_test_scaled = scaler_X.transform(X_test_np)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_np)
    y_test_scaled = scaler_y.transform(y_test_np)
    X_train_nnf, Y_train_nnf = convert_to_nnf_format(X_train_scaled, y_train_scaled)
    X_test_nnf, Y_test_nnf = convert_to_nnf_format(X_test_scaled, y_test_scaled)
    n_features = X_train_np.shape[1]
    

    model = Model([
        Dense(n_features, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 1)
    ])
    trainer = Trainer(model=model, loss_fn=MSELoss(), optimizer=Adam(lr=0.01), epochs=100)
    trainer.fit(X_train_nnf, Y_train_nnf, verbose=True) 
    predictions_scaled_nnf = trainer.predict(X_test_nnf)
    pred_scaled_list = [p.data[0][0] for p in predictions_scaled_nnf]
    pred_scaled_np = np.array(pred_scaled_list).reshape(-1, 1)
    predictions_actual = scaler_y.inverse_transform(pred_scaled_np)

if __name__ == "__main__":
    random.seed(42) 
    np.random.seed(42)
    main()

