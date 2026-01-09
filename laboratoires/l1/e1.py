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
    num_epochs = int(100_000)
    learning_rates = (0.005, 0.001, 0.01)
    # NOTE: 0.01 results in `nan` -> too high a rate.

    for i, learning_rate in enumerate(learning_rates, start=1):
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        Bs.append(B_star)

        fig = plt.figure(i, figsize=(8, 6))
        ax = fig.add_subplot()
        ax.semilogx(epochs, losses)
        ax.set_title(f"Matrice 3x3, pas: {learning_rate}")
        ax.set_xlabel("époque")
        ax.set_ylabel("L")
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
    num_epochs = 1_000_000
    learning_rates = (0.001, 0.0001)
    # NOTE:
    #   - 0.005 results in `nan` -> too high a learning rate.
    #   - 10'000 and 100'000 epochs: too low to reach adequate results.
    #   - 1M epochs too low for 0.0001 learning rate.

    for i, learning_rate in enumerate(learning_rates, start=4):
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        B6s.append(B_star)

        fig = plt.figure(i, figsize=(8, 6))
        ax = fig.add_subplot()
        ax.semilogx(epochs, losses)
        ax.set_title(f"Matrice 6x6, pas: {learning_rate}")
        ax.set_xlabel("époque")
        ax.set_ylabel("L")
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
    num_epochs = 100_000
    learning_rates = (0.001, 0.0001)

    for i, learning_rate in enumerate(learning_rates, start=6):
        B = np.random.rand(*A.shape).astype(A.dtype)
        B_star, epochs, losses = train(A, B, learning_rate, num_epochs)
        B4s.append(B_star)

        fig = plt.figure(i, figsize=(8, 6))
        ax = fig.add_subplot()
        ax.semilogx(epochs, losses)
        ax.set_title(f"Matrice 4x4 (Singulière), pas: {learning_rate}")
        ax.set_xlabel("époque")
        ax.set_ylabel("L")
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
