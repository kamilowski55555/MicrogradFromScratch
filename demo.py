"""
Simple demo of micrograd: training a tiny neural network on a simple dataset.
"""
from micrograd import Value, MLP
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    print("=" * 60)
    print("Micrograd Demo: Training a Simple Neural Network")
    print("=" * 60)
    print()

    # Example 1: Basic operations with Value
    print("1. Basic automatic differentiation:")
    print("-" * 40)
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b
    c.label = 'c'
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    print(f"c = a * b = {c.data}")

    c.backward()
    print(f"dc/da = {a.grad}")
    print(f"dc/db = {b.grad}")
    print()

    # Example 2: More complex expression
    print("2. Complex expression: f(x,y) = x*y + (x+y)**2")
    print("-" * 40)
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    f = x * y + (x + y)**2
    f.label = 'f'

    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"f = {f.data}")

    f.backward()
    print(f"df/dx = {x.grad}")
    print(f"df/dy = {y.grad}")
    print()

    # Example 3: Training a neural network
    print("3. Training a neural network on XOR-like problem:")
    print("-" * 40)

    # Create a simple dataset (XOR-like)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

    # Create a neural network: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    model = MLP(3, [4, 4, 1])
    print(f"Created MLP with {len(model.parameters())} parameters")
    print()

    # Training loop
    print("Training progress:")
    learning_rate = 0.05
    epochs = 50

    for epoch in range(epochs):
        # Forward pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Update parameters
        for p in model.parameters():
            p.data += -learning_rate * p.grad

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

    print()
    print("Final predictions vs targets:")
    print("-" * 40)
    ypred = [model(x) for x in xs]
    for i, (x, target, pred) in enumerate(zip(xs, ys, ypred)):
        print(f"Input {i+1}: {[round(xi, 2) for xi in x]} -> "
              f"Target: {target:6.2f} | Prediction: {pred.data:6.3f}")

    print()

    # Example 4: Real dataset from sklearn
    print("4. Training on sklearn's make_moons dataset:")
    print("-" * 40)

    # Generate a binary classification dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    y = y * 2 - 1  # Convert labels from {0,1} to {-1,1}

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features per sample: {X_train.shape[1]}")
    print()

    # Create a neural network: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
    model_real = MLP(2, [16, 16, 1])
    print(f"Created MLP with {len(model_real.parameters())} parameters")
    print()

    # Training loop
    print("Training progress:")
    learning_rate = 0.01
    epochs = 100

    for epoch in range(epochs):
        # Forward pass on training data
        ypred = [model_real(x.tolist()) for x in X_train]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(y_train, ypred))

        # Zero gradients
        for p in model_real.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Update parameters
        for p in model_real.parameters():
            p.data += -learning_rate * p.grad

        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            # Calculate accuracy on training data
            correct = sum((yout.data > 0) == (ygt > 0) for ygt, yout in zip(y_train, ypred))
            accuracy = correct / len(y_train) * 100
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f} | Train Accuracy: {accuracy:.1f}%")

    print()
    print("Test Set Evaluation:")
    print("-" * 40)

    # Evaluate on test set
    ypred_test = [model_real(x.tolist()) for x in X_test]
    test_correct = sum((yout.data > 0) == (ygt > 0) for ygt, yout in zip(y_test, ypred_test))
    test_accuracy = test_correct / len(y_test) * 100

    print(f"Test Accuracy: {test_accuracy:.1f}% ({test_correct}/{len(y_test)} correct)")
    print()

    # Show some predictions
    print("Sample predictions (first 5 test samples):")
    for i in range(min(5, len(X_test))):
        pred_val = ypred_test[i].data
        pred_class = 1 if pred_val > 0 else -1
        actual = y_test[i]
        status = "✓" if pred_class == actual else "✗"
        print(f"  {status} Input: [{X_test[i][0]:.2f}, {X_test[i][1]:.2f}] -> "
              f"Predicted: {pred_class:2d} (score: {pred_val:6.3f}) | Actual: {actual:2d}")

    print()
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

