import numpy as np
from typing import Any
from dnn_framework.common import Array
from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer (linear layer).
    """

    def __init__(self, input_count: int, output_count: int):
        super().__init__()
        self.input_count = input_count
        self.output_count = output_count
        self.W = np.random.normal(
            0, 2 / (input_count + output_count), (output_count, input_count)
        )  # dim(J, I)
        self.b = np.random.normal(0, 2 / (input_count + output_count), output_count)  # dim(J, 1)
        assert self.W.ndim == 2
        assert self.b.ndim == 1
        assert self.W.shape[0] == self.b.shape[0]

    def get_parameters(self) -> dict[str, Any]:
        return {"w": self.W, "b": self.b}

    def get_buffers(self):
        return {}

    def forward(self, x: Array) -> tuple[Array, dict[str, Any]]:
        """
        Inférence d'une couche linéaire.

        Arguments:
            x: Matrice des entrées. dim(N, I).
        Implicit:
            W: Matrice des poids. dim(J, I).
            b: Matrice des biais. dim(J, 1).
        Returns:
            out: Matrice `y_hat`. dim(N, J).
        """
        return ((x @ self.W.T + np.tile(self.b, (x.shape[0], 1))).astype(x.dtype), {"x": x})

    def backward(self, output_grad: Array, cache: dict[str, Any]) -> tuple[Array, dict[str, Any]]:
        """
        Rétropropagation du gradient d'une couche linéaire.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, J).
            cache: Matrice des entrées. dim(N, I).
        Returns:
            out: Gradients dL/dX (N x I), dL/dW (J x I), dL/db (J x 1).
        """
        assert self.W.shape[0] == output_grad.shape[1]
        assert self.W.shape[1] == cache["x"].shape[1]
        assert cache["x"].shape[0] == output_grad.shape[0]
        dLdX = (output_grad @ self.W).astype(cache["x"].dtype)
        dLdW = (output_grad.T @ cache["x"]).astype(cache["x"].dtype)
        dLdb = np.sum(output_grad, axis=0, dtype=cache["x"].dtype)
        return (dLdX, {"w": dLdW, "b": dLdb})


class BatchNormalization(Layer):
    """This class implements a batch normalization layer."""

    def __init__(
        self,
        input_count: int,
        alpha: float = 0.1,
        epsilon: float = 1e-9,
    ):
        assert 0 <= alpha <= 1
        super().__init__()
        self.input_count = input_count
        self.alpha = alpha
        self.gamma = np.zeros((input_count,))
        self.beta = np.zeros((input_count,))
        self.epsilon = epsilon
        self.global_mean = np.zeros((input_count,))
        self.global_variance = np.ones((input_count,))

    def get_parameters(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "beta": self.beta}

    def get_buffers(self) -> dict[str, Any]:
        return {"global_mean": self.global_mean, "global_variance": self.global_variance}

    def forward(self, x: Array) -> tuple[Array, dict[str, Any]]:
        if self._is_training:
            return self._forward_training(x)
        else:
            return (self._forward_evaluation(x), {})

    def _forward_training(self, x: Array) -> tuple[Array, dict[str, Any]]:
        batch_mean = np.mean(x, axis=0, keepdims=True)
        batch_variance = np.var(x, axis=0, keepdims=True)

        # Update global parameters (for evaluation)
        self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * batch_mean
        self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * batch_variance

        # Inference
        xhat = (x - batch_mean) / np.sqrt(batch_variance + self.epsilon)
        yhat = (self.gamma * xhat + self.beta).astype(x.dtype)

        return (yhat, {"x": x, "xhat": xhat, "batch_mean": batch_mean, "batch_var": batch_variance})

    def _forward_evaluation(self, x: Array) -> Array:
        xhat = (x - self.global_mean) / np.sqrt(self.global_variance + self.epsilon)
        return self.gamma * xhat + self.beta

    def backward(self, output_grad: Array, cache: dict[str, Any]) -> tuple[Array, dict[str, Any]]:
        dLdx_hat = output_grad * self.gamma
        dLdvar_B = np.sum(
            dLdx_hat
            * (cache["x"] - cache["batch_mean"])
            * (-0.5 * np.pow(cache["batch_var"] + self.epsilon, -3 / 2)),
            axis=0,
        )
        batch_stddev = np.sqrt(cache["batch_var"] + self.epsilon)
        dLdmean_B = -np.sum(dLdx_hat / batch_stddev, axis=0)
        M = output_grad.size
        dLdx = (
            dLdx_hat / batch_stddev
            + 2 / M * dLdvar_B * (cache["x"] - cache["batch_mean"])
            + 1 / M * dLdmean_B
        )
        dLdgamma = np.sum(output_grad * cache["xhat"], axis=0)
        dLdbeta = np.sum(output_grad, axis=0)
        return (dLdx, {"gamma": dLdgamma, "beta": dLdbeta})


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x: Array) -> tuple[Array, dict[str, Array]]:
        y = (1 / (1 + np.exp(-x))).astype(x.dtype)
        return (y, {"y": y})

    def backward(self, output_grad: Array, cache: dict[str, Any]) -> tuple[Array, dict[str, Any]]:
        """
        Rétropropagation du gradient.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, I).
            cache: Sortie estimée lors de l'inférence `y_hat`. dim(N, I).
        Returns:
            out: Gradient dL/dx. dim(N, I).
        """
        dLdx = ((1 - cache["y"]) * cache["y"] * output_grad).astype(cache["y"].dtype)
        return (dLdx, {})


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x: Array) -> tuple[Array, dict[str, Any]]:
        y_hat = np.where(x >= 0, x, 0).astype(x.dtype)
        return (y_hat, {"x": x})

    def backward(self, output_grad: Array, cache: dict[str, Any]):
        """
        Rétropropagation du gradient.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, I).
            cache: Matrice des entrées. dim(N, I).
        Returns:
            out: Gradient dL/dx. dim(N, I).
        """
        return (
            (np.where(cache["x"] >= 0, 1, 0) * output_grad).astype(cache["x"].dtype),
            {},
        )
