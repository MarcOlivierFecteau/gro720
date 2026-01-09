#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
Array1D = np.ndarray[tuple[int], np.dtype[np.float32]]

data: Array2D = np.array(
    [
        (-0.95, 0.02),
        (-0.82, 0.03),
        (-0.62, -0.17),
        (-0.43, -0.12),
        (-0.17, -0.37),
        (-0.07, -0.25),
        (0.25, -0.1),
        (0.38, 0.14),
        (0.61, 0.53),
        (0.79, 0.71),
        (1.04, 1.53),
    ],
    dtype=np.float32,
)


def build_polynomial_matrix(x: Array1D, order: int) -> Array2D:
    """Builds the polynomial design matrix [1, x, x^2, ..., x^N]."""
    assert order >= 0
    return np.column_stack([x**i for i in range(order + 1)])


def loss(y_hat: Array1D, y: Array1D) -> float:
    """MSE loss."""
    return float(np.sum((y_hat - y) ** 2, dtype=y.dtype))


# NOTE: Unused.
def gradient_single(x_poly: Array2D, w: Array1D, y: Array1D, order: int) -> float:
    """dL/da_n."""
    assert 0 <= order < y.shape[0]
    y_hat = w * x_poly[:, order]
    return 2 * np.sum((y_hat - y) @ x_poly[:, order])


def gradient(x_poly: Array2D, w: Array1D, y: Array1D) -> Array1D:
    """dL/da."""
    y_hat = w @ x_poly.T
    return 2 * np.sum((y_hat - y) @ x_poly, axis=0)


def update_weights(x_poly: Array2D, w: Array1D, y: Array1D, learning_rate: float) -> Array1D:
    return w - (learning_rate * gradient(x_poly, w, y)).astype(w.dtype)


def train(
    x_poly: Array2D, y: Array1D, w: Array1D, learning_rate: float, num_epochs: int
) -> tuple[Array1D, list[int], list[float]]:
    assert x_poly.shape[1] == w.shape[0]
    losses: list[float] = []
    epochs = list(range(1, num_epochs + 1))
    for _ in tqdm(epochs):
        y_hat = w @ x_poly.T
        losses.append(loss(y_hat, y))
        w = update_weights(x_poly, w, y, learning_rate)
    return (w, epochs, losses)


def polynomial_regression(
    order: int,
    x: Array1D,
    y: Array1D,
    x_validation: Array1D,
    num_epochs: int,
    learning_rates: list[float],
) -> None:
    print(f"Régression Polynomiale, ordre {order}:\n")
    x_poly = build_polynomial_matrix(x, order)
    x_validation_poly = build_polynomial_matrix(x_validation, order)
    epochses: list[list[int]] = []
    losseses: list[list[float]] = []
    predictions: list[Array1D] = []

    for _, learning_rate in enumerate(learning_rates):
        w: Array1D = np.random.rand(order + 1).astype(np.float32)
        w_star, epochs, losses = train(x_poly, y, w, learning_rate, num_epochs)
        prediction = w_star @ x_validation_poly.T
        predictions.append(prediction)
        epochses.append(epochs)
        losseses.append(losses)

    fig = plt.figure(order, figsize=(8, 6))
    fig.suptitle(f"Régression Polynomiale, ordre: {order}")
    ax = fig.add_subplot(2, 1, 1)
    for epochs, losses, learning_rate in zip(epochses, losseses, learning_rates):
        ax.semilogx(epochs, losses, label=f"mu={learning_rate}")
    ax.set_xlabel("époque")
    ax.set_ylabel("L")
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(x, y, label="Entraînement")
    for prediction, learning_rate in zip(predictions, learning_rates):
        ax.plot(x_validation, prediction, label=f"mu={learning_rate}")
    ax.set_title("Validation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    fig.tight_layout()


def main() -> None:
    np.random.seed(42)
    x: Array1D = data[:, 0]
    y: Array1D = data[:, 1]
    x_validation = np.linspace(-1.25, 1.25, 100, dtype=np.float32)

    # NOTE: 1000 epochs is good enough for learning rates [0.0001, 0.001] up to order 7 polynomial regression.

    # NOTE: Linear regression
    #   - Learning rate of 0.0001 gives poor performance (too low for linear regression)
    #   - Under-fitting observed
    polynomial_regression(1, x, y, x_validation, 1_000, [0.001, 0.005, 0.01])

    # NOTE: Quadratic regression
    #   - Learning rate of 0.01 resuts in poor performance
    #   - 1000 epochs is too short -> 10'000 is adequate
    #   - Learning rate of 0.0001 gives poor performance (too low for quadratic regression)
    #   - Learning rate of 0.001 gives best performance (by far)
    #   - Under-fitting observed
    polynomial_regression(2, x, y, x_validation, 1_000, [0.001, 0.0001, 0.01])

    for order in range(3, 7):
        # NOTE: Outside scope of exercise; for observation purposes.
        polynomial_regression(order, x, y, x_validation, 1_000, [0.005, 0.001, 0.0005, 0.0001])

    # NOTE: Order 7 polynomial regression
    #   - Learning rate of 0.01 is too low
    #   - 10'000 epochs is too short -> 100'000 is adequate
    #   - Slight over-fitting observed
    polynomial_regression(7, x, y, x_validation, 1_000, [0.001, 0.0001, 0.0005])

    plt.show(block=True)


if __name__ == "__main__":
    main()
