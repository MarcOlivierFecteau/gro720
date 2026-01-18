import numpy as np

from dnn_framework.common import Array, Float
from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x: Array, target: Array) -> tuple[Float, Array]:
        """
        N: Number of elements in the batch.\\
        I: Number of caracteristics in the input vector.
        Arguments:
            x: Matrice de la sortie estimée `y_hat`. dim(N, I).
            target: Matrice des cibles. dim(N, 1).
        Returns:
            (loss, dLdx): Coût (perte) moyen (1 x 1) et dL/dx (N x I).
        """
        assert x.shape[0] == target.shape[0]
        EPSILON = 1e-9
        sample_count: int = x.shape[0]
        yhat = softmax(x)

        # One-hot encode target
        target_one_hot = np.zeros_like(yhat)
        target_one_hot[np.arange(sample_count), target] = 1  # type: ignore
        loss = np.sum(-np.sum(target_one_hot * np.log(yhat + EPSILON), axis=1)) / sample_count
        dLdx = (yhat - target_one_hot) / sample_count
        # NOTE: ^^^ Not sure how we obtain that expression (got it through Claude Haiku 4.5)
        return (loss, dLdx)


def softmax(x: Array) -> Array:
    """
    Arguments:
        x: Matrice des entrées. dim(N, I).
    Returns:
        out: Matrice des valeurs projetées. dim(N, 1).
    """
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x: Array, target: Array) -> tuple[Float, Array]:
        """
        Arguments:
            x: Matrice de la sortie estimée `y_hat`. dim(N, 1).
            target: Matrice des cibles. dim(N, 1).
        Returns:
            (loss, dLdx): Coût (perte) moyen (1 x 1) et dL/dx (N x 1).
        """
        assert x.shape[0] == target.shape[0]
        sample_count: int = x.shape[0]
        output_count: int = x.shape[1]
        loss = np.sum(np.sum((x - target) ** 2, axis=1) / output_count) / sample_count
        dLdx = 2 * (x - target) / (output_count * sample_count)
        return (loss, dLdx)
