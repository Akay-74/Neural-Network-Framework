import NNF

def main():
    print("=== Neural Network Framework Test ===")

    # Matrix operations
    A = NNF.Matrix(2, 2, 1.0)
    B = NNF.Matrix(2, 2, 2.0)
    print("Matrix A:", A.data)
    print("Matrix B:", B.data)
    print("A + B =", (A + B).data)
    print("A - B =", (A - B).data)
    print("A @ B =", (A @ B).data)

    # Build a simple model
    model = NNF.Sequential(
        NNF.Linear(2, 4),
        NNF.ReLU(),
        NNF.Linear(4, 2),
        NNF.Sigmoid()
    )

    # Dummy dataset
    X = [NNF.Matrix(2, 1, i) for i in [0.5, 1.0, 1.5, 2.0]]
    y = [NNF.Matrix(2, 1, j) for j in [0, 1, 0, 1]]
    dataset = list(zip(X, y))

    # Loss & Trainer
    loss_fn = NNF.MSELoss()
    trainer = NNF.Trainer(model, loss_fn, lr=0.01, epochs=200)

    # Train
    print("\n=== Training ===")
    trainer.fit(dataset, verbose=True)

    # Predictions
    print("\n=== Predictions ===")
    preds = trainer.predict(X)
    for i, (x, pred) in enumerate(zip(X, preds)):
        print(f"Input {i+1}: {x.data} -> Predicted: {pred.data}")

    # Evaluation
    print("\n=== Evaluation ===")
    final_loss = trainer.evaluate(dataset)
    print("Final Loss:", final_loss)

if __name__ == "__main__":
    main()