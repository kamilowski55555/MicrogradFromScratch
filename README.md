# Micrograd - Learning from Scratch

My attempt at replicating **micrograd** from Andrej Karpathy's ["Neural Networks: Zero to Hero"](https://www.youtube.com/watch?v=VMj-3S1tku0) course.

This is a minimal automatic differentiation engine with a tiny neural network library built on top. The goal is to understand how backpropagation and neural networks work at a fundamental level by building them from scratch.

## What's Inside

```
micrograd/
├── engine.py          # Value class with automatic differentiation
├── nn.py              # Neuron, Layer, and MLP classes
└── __init__.py        

demo.py                # Demo with toy data and sklearn datasets
micrograd_from_scratch.ipynb  # Original learning notebook
```

## Quick Example

```python
from micrograd import Value

# Automatic differentiation
a = Value(2.0)
b = Value(3.0)
c = a * b + a**2

c.backward()
print(f"dc/da = {a.grad}")  # 7.0
print(f"dc/db = {b.grad}")  # 2.0
```

## Training a Neural Network

```python
from micrograd import MLP

# Create network: 3 inputs -> 4 -> 4 -> 1 output
model = MLP(3, [4, 4, 1])

# Training data
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Simple training loop
for epoch in range(100):
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    for p in model.parameters():
        p.grad = 0.0
    
    loss.backward()
    
    for p in model.parameters():
        p.data += -0.05 * p.grad
```

## Run the Demo

```bash
python demo.py
```

The demo includes:
- Basic automatic differentiation examples
- Training on toy XOR-like problem
- Training on sklearn's `make_moons` dataset (real binary classification)

## Requirements

- Python 3.7+
- scikit-learn (for real dataset demo)

## Credit

Based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his excellent [video tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0).

