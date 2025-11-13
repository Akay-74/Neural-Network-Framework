"""
NNF vs TensorFlow (Short Version) – 100 Epochs
NOW WITH LIVE EPOCH PROGRESS PRINTING
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import random, time

# Import NNF
from NNF import Tensor, Dense, ReLU, Softmax, Model, CrossEntropyLoss, Adam


# -------------------- DATA --------------------
iris = load_iris()
X = iris.data
y = iris.target
n_classes = 3

# Preserve unscaled test data
X_train_orig, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_orig)
X_test  = scaler.transform(X_test_original)

# One-hot encode
y_train_oh = np.eye(n_classes)[y_train]
y_test_oh  = np.eye(n_classes)[y_test]

# Convert to NNF format
X_train_nnf = [Tensor([[float(v)] for v in sample]) for sample in X_train]
X_test_nnf  = [Tensor([[float(v)] for v in sample]) for sample in X_test]
y_train_nnf = [Tensor([[float(v)] for v in sample]) for sample in y_train_oh]


# -------------------- NNF MODEL --------------------
nnf_model = Model([
    Dense(4, 16), ReLU(),
    Dense(16, 8), ReLU(),
    Dense(8, 3), Softmax()
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(lr=0.01)

nnf_losses = []
nnf_accuracies = []
epochs = 100

print("\n----- Training NNF (Live Epochs) -----")
start_time = time.time()

for epoch in range(1, epochs + 1):
    total_loss = 0

    order = list(range(len(X_train_nnf)))
    random.shuffle(order)

    for i in order:
        x = X_train_nnf[i]
        y_true = y_train_nnf[i]

        y_pred = nnf_model.forward(x)
        loss = loss_fn.forward(y_pred, y_true)
        total_loss += loss

        grad = loss_fn.backward()
        nnf_model.backward(grad)

        params = nnf_model.get_params()
        nnf_model.set_params(optimizer.step(params))

    # Accuracy on test
    preds = [nnf_model.forward(x) for x in X_test_nnf]
    pred_classes = np.array([np.argmax([v[0] for v in p.data]) for p in preds])
    acc = accuracy_score(y_test, pred_classes)

    nnf_losses.append(total_loss / len(X_train_nnf))
    nnf_accuracies.append(acc)

    # LIVE EPOCH PRINT
    print(f"NNF Epoch {epoch}/{epochs} | Loss: {nnf_losses[-1]:.4f} | Acc: {acc:.4f}")

print(f"NNF Training Time: {time.time() - start_time:.2f}s")


# -------------------- TENSORFLOW MODEL --------------------
tf_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
tf_model.compile(optimizer=keras.optimizers.Adam(0.01),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

print("\n----- Training TensorFlow (Live Epochs) -----")

start_time = time.time()

# Custom training loop to show live epochs
tf_losses = []
tf_accuracies = []

for epoch in range(1, epochs + 1):
    history = tf_model.fit(
        X_train, y_train_oh,
        epochs=1,           # train exactly ONE epoch
        batch_size=1,
        validation_data=(X_test, y_test_oh),
        verbose=0
    )

    loss = history.history['loss'][0]
    acc  = history.history['val_accuracy'][0]

    tf_losses.append(loss)
    tf_accuracies.append(acc)

    print(f"TF  Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

print(f"TF Training Time: {time.time() - start_time:.2f}s")


# -------------------- PLOTS --------------------
# LOSS PLOT
plt.figure(figsize=(10,4))
plt.plot(nnf_losses, label="NNF Loss")
plt.plot(tf_losses, label="TF Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ACCURACY PLOT
plt.figure(figsize=(10,4))
plt.plot(nnf_accuracies, label="NNF Accuracy")
plt.plot(tf_accuracies, label="TF Accuracy")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# -------------------- ACTUAL VS PREDICTED --------------------
def plot_actual_vs_predicted(X_test_original, y_test, nnf_pred, tf_pred, feature_names, feature_index=0):
    feat = X_test_original[:, feature_index]
    name = feature_names[feature_index]

    idx = np.argsort(feat)
    Xs = feat[idx]
    ya = y_test[idx]
    nnf_s = nnf_pred[idx]
    tf_s = tf_pred[idx]

    plt.figure(figsize=(12,6))
    plt.plot(Xs, ya, "o-", label="Actual", alpha=0.7)
    plt.plot(Xs, nnf_s, "x--", label="NNF Predicted", alpha=0.7)
    plt.plot(Xs, tf_s, "s--", label="TF Predicted", alpha=0.7)

    plt.title(f"Actual vs Predicted vs Feature: {name}")
    plt.xlabel(name)
    plt.ylabel("Class (0,1,2)")
    plt.yticks([0,1,2])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# Final predictions
tf_pred_probs = tf_model.predict(X_test, verbose=0)
tf_pred_classes = np.argmax(tf_pred_probs, axis=1)

plot_actual_vs_predicted(
    X_test_original,
    y_test,
    pred_classes,
    tf_pred_classes,
    iris.feature_names,
    feature_index=0
)

print("\nDONE.")
