#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]]


def loss(X: Array2D) -> float:
    return float(np.sum(X**2, dtype=X.dtype))


def gradient(A: Array2D, B: Array2D) -> Array2D:
    return 2 * (B @ A - np.eye(*A.shape, dtype=A.dtype)) @ A.T


def estimate_B(A: Array2D, B: Array2D, learning_rate: float) -> Array2D:
    return B - learning_rate * gradient(A, B)


def train(
    A: Array2D, B: Array2D, learning_rate: float, num_epochs: int
) -> tuple[Array2D, list[int], list[float]]:
    assert A.shape == B.shape
    _I = np.eye(*A.shape, dtype=A.dtype)
    losses: list[float] = []
    epochs = list(range(1, num_epochs + 1))
    for _ in tqdm(epochs):
        result = B @ A - _I
        losses.append(loss(result))
        B = estimate_B(A, B, learning_rate)
    return (B, epochs, losses)


def main() -> None:
    print("Matrice 3x3:\n")
    A = np.array(
        [
            [3, 4, 1],
            [5, 2, 3],
            [6, 2, 2],
        ],
        dtype=np.float32,
    )
    Bs: list[Array2D] = []
    epochses: list[list[int]] = []
    losseses: list[list[float]] = []

    num_epochs = int(10_000)
    learning_rates = (0.005, 0.001, 0.01)
    # NOTE: 0.01 results in `nan` -> too high a rate.

    for learning_rate in learning_rates:
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        Bs.append(B_star)
        epochses.append(epochs)
        losseses.append(losses)

    fig = plt.figure(3, figsize=(8, 6))
    ax = fig.add_subplot()
    for epochs, losses, learning_rate in zip(epochses, losseses, learning_rates):
        ax.loglog(epochs, losses, label=f"mu={learning_rate}")
    ax.set_title("Matrice 3x3")
    ax.set_xlabel("époque")
    ax.set_ylabel("L")
    ax.legend()
    fig.tight_layout()

    # Display results
    print("=" * 30)
    for B, learning_rate in zip(Bs, learning_rates):
        print(f"Step = {learning_rate}")
        print(f"A_inv (B):\n{B}")
        print("=" * 30)

    print("\nMatrice 6x6:\n")
    A = np.array(
        [
            [3, 4, 1, 2, 1, 5],
            [5, 2, 3, 2, 2, 1],
            [6, 2, 2, 6, 4, 5],
            [1, 2, 1, 3, 1, 2],
            [1, 5, 2, 3, 3, 3],
            [1, 2, 2, 4, 2, 1],
        ],
        dtype=np.float32,
    )
    B6s: list[Array2D] = []
    epochses: list[list[int]] = []
    losseses: list[list[float]] = []

    num_epochs = 500_000
    learning_rates = (0.001, 0.0005, 0.0001)
    # NOTE:
    #   - 0.005 results in `nan` -> too high a learning rate.
    #   - 10'000 and 100'000 epochs too low to reach adequate results.
    #   - 1M epochs too low for 0.0001 learning rate. -> 0.0001 too low a rate for 6x6 matrix.

    for learning_rate in learning_rates:
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        B6s.append(B_star)
        epochses.append(epochs)
        losseses.append(losses)

    fig = plt.figure(6, figsize=(8, 6))
    ax = fig.add_subplot()
    for epochs, losses, learning_rate in zip(epochses, losseses, learning_rates):
        ax.loglog(epochs, losses, label=f"mu={learning_rate}")
    ax.set_title("Matrice 6x6")
    ax.set_xlabel("époque")
    ax.set_ylabel("L")
    ax.legend()
    fig.tight_layout()

    # Display results
    print("=" * 30)
    for B, learning_rate in zip(B6s, learning_rates):
        print(f"Step = {learning_rate}")
        print(f"A_inv (B):\n{B}")
        print("=" * 30)
    print("Analytic Solution:")
    print(np.linalg.inv(A))

    print("\nMatrice 4x4 (Singulière):\n")
    A = np.array(
        [[2, 1, 1, 2], [1, 2, 3, 2], [2, 1, 1, 2], [3, 1, 4, 1]],
        dtype=np.float32,
    )  # NOTE: Singular matrix.
    B4s: list[Array2D] = []
    epochses: list[list[int]] = []
    losseses: list[list[float]] = []

    num_epochs = 10_000
    learning_rates = (0.001, 0.0005, 0.0001)

    for learning_rate in learning_rates:
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        B4s.append(B_star)
        epochses.append(epochs)
        losseses.append(losses)

    fig = plt.figure(4, figsize=(8, 6))
    ax = fig.add_subplot()
    for epochs, losses, learning_rate in zip(epochses, losseses, learning_rates):
        ax.loglog(epochs, losses, label=f"mu={learning_rate}")
    ax.set_title("Matrice 4x4 (Singulière)")
    ax.set_xlabel("époque")
    ax.set_ylabel("L")
    ax.legend()
    fig.tight_layout()
    # NOTE: Linear regression model does NOT know that A in singular, and
    #       still tries to find its inverse.

    # Display results
    print("=" * 30)
    for B, learning_rate in zip(B4s, learning_rates):
        print(f"Step = {learning_rate}")
        print(f"A_inv (B):\n{B}")
        print("=" * 30)

    plt.show(block=True)


if __name__ == "__main__":
    main()
