import numpy as np

from dnn_framework.common import Array, Float
from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer (linear layer).
    """

    def __init__(self, input_count: int, output_count: int):
        self.input_count = input_count
        self.output_count = output_count
        self.W = np.random.normal(
            0, 2 / (input_count + output_count), (output_count, input_count)
        )  # dim(J, I)
        self.b = np.random.normal(0, 2 / (input_count + output_count), output_count)  # dim(J, 1)
        assert self.W.ndim == 2
        assert self.b.ndim == 1
        assert self.W.shape[0] == self.b.shape[0]

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x: Array) -> Array:
        """
        Inférence d'une couche linéaire.

        Arguments:
            x: Matrice des entrées. dim(N, I).
        Returns:
            out: Matrice `y_hat`. dim(N, J).
        """
        return (x @ self.W.T + np.tile(self.b, (x.shape[0], 1))).astype(x.dtype)

    def backward(self, output_grad: Array, cache: Array) -> tuple[Array, Array, Float | Array]:
        """
        Rétropropagation du gradient d'une couche linéaire.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, J).
            cache: Matrice des entrées. dim(N, I).
        Returns:
            out: Gradients dL/dX (N x I), dL/dW (J x I), dL/db (J x 1).
        """
        assert self.W.shape[0] == output_grad.shape[1]
        assert self.W.shape[1] == cache.shape[1]
        assert cache.shape[0] == output_grad.shape[0]
        dLdX = (output_grad @ self.W).astype(cache.dtype)
        dLdW = (output_grad.T @ cache).astype(cache.dtype)
        dLdb = np.sum(output_grad, axis=0, dtype=cache.dtype)
        return (dLdX, dLdW, dLdb)


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(
        self,
        input_count: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        beta: float = 0.0,
        epsilon: float = 1e-9,
    ):
        assert 0 <= alpha <= 1
        self.N = input_count
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

    def get_parameters(self) -> tuple[int, float, float, float, float]:
        return (self.N, self.alpha, self.gamma, self.beta, self.epsilon)

    def get_buffers(self):
        raise NotImplementedError("BN.get_buffers")

    def forward(self, x: Array) -> Array:
        mean = np.mean(x)
        variance = np.var(x)
        return ((x - mean) / (variance + self.epsilon)).astype(x.dtype)

    def _forward_training(self, x: Array):
        raise NotImplementedError("BN._forward_training")

    def _forward_evaluation(self, x: Array):
        raise NotImplementedError("BN._forward_evaluation")

    def backward(self, output_grad: Array, cache: Array):
        raise NotImplementedError("BN.backward")


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        pass

    def get_buffers(self):
        raise NotImplementedError("Sigmoid.get_buffers")

    def forward(self, x: Array) -> Array:
        return (1 / (1 + np.exp(-x))).astype(x.dtype)

    def backward(self, output_grad: Array, cache: Array) -> Array:
        """
        Rétropropagation du gradient.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, I).
            cache: Sortie estimée lors de l'inférence. dim(N, I).
        Returns:
            out: Gradient dL/dx. dim(N, I).
        """
        return ((1 - cache) * cache * output_grad).astype(cache.dtype)


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        pass

    def get_buffers(self):
        raise NotImplementedError("ReLU.get_buffers")

    def forward(self, x: Array):
        return np.where(x >= 0, x, 1).astype(x.dtype)

    def backward(self, output_grad: Array, cache: Array):
        """
        Rétropropagation du gradient.

        Arguments:
            output_grad: Gradient de la couche en aval. dim(N, I).
            cache: Matrice des entrées. dim(N, I).
        Returns:
            out: Gradient dL/dx. dim(N, I).
        """
        return (np.where(cache >= 0, 1, 0) * output_grad).astype(cache.dtype)
