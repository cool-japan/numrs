#!/usr/bin/env python3
"""
NumRS2 Neural Networks Example

Demonstrates neural network primitives including activation functions,
loss functions, and normalization.
"""

import numrs2 as nr


def main():
    print("NumRS2 Neural Networks Examples")
    print("=" * 60)
    print()

    # Activation functions
    print("1. Activation Functions")
    print("-" * 60)

    # ReLU
    x = nr.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"Input: {x.tolist()}")

    relu_out = nr.nn.relu(x)
    print(f"ReLU: {relu_out.tolist()}")

    # Sigmoid
    sigmoid_out = nr.nn.sigmoid(x)
    print(f"Sigmoid: {sigmoid_out.tolist()}")

    # Tanh
    tanh_out = nr.nn.tanh(x)
    print(f"Tanh: {tanh_out.tolist()}")
    print()

    # Softmax
    print("2. Softmax")
    print("-" * 60)

    logits = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Logits: {logits.tolist()}")

    probs = nr.nn.softmax(logits)
    print(f"Softmax probabilities: {probs.tolist()}")
    print(f"Sum of probabilities: {probs.sum()}")
    print()

    # Loss functions
    print("3. Loss Functions")
    print("-" * 60)

    # MSE Loss
    predictions = nr.array([1.5, 2.3, 3.1, 4.2, 5.0])
    targets = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

    print(f"Predictions: {predictions.tolist()}")
    print(f"Targets: {targets.tolist()}")

    mse = nr.nn.mse_loss(predictions, targets)
    print(f"MSE Loss: {mse}")
    print()

    # Dropout
    print("4. Dropout")
    print("-" * 60)

    x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    print(f"Input: {x.tolist()}")

    # Dropout with p=0.5
    dropped = nr.nn.dropout(x, 0.5)
    print(f"After dropout (p=0.5): {dropped.tolist()}")
    print()

    # Batch normalization
    print("5. Batch Normalization")
    print("-" * 60)

    batch = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Batch: {batch.tolist()}")
    print(f"Mean: {batch.mean()}")

    normalized = nr.nn.batch_norm(batch)
    print(f"After batch norm: {normalized.tolist()}")
    print(f"Normalized mean: {normalized.mean()}")
    print()

    # Simple forward pass example
    print("6. Simple Forward Pass")
    print("-" * 60)

    print("Simulating a simple neural network layer:")
    print("Input -> Linear -> ReLU -> Output")
    print()

    # Input
    input_vec = nr.array([0.5, -0.3, 0.8])
    print(f"Input: {input_vec.tolist()}")

    # Simulate linear transformation (for demo, just multiply by 2)
    linear_out = input_vec * nr.array([2.0, 2.0, 2.0])
    print(f"After linear: {linear_out.tolist()}")

    # Apply ReLU
    output = nr.nn.relu(linear_out)
    print(f"After ReLU: {output.tolist()}")
    print()

    print("=" * 60)
    print("Neural network examples completed successfully!")


if __name__ == "__main__":
    main()
