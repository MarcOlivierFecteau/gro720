import numpy as np

from dnn_framework.common import Array, Float
from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x: Array, target: Array) -> tuple[Float, Array]:
        """
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
        loss = -np.sum(target_one_hot * np.log(yhat + EPSILON), axis=0)
        dLdx = (yhat - target_one_hot).astype(x.dtype) / sample_count
        # NOTE: ^^^ Not sure how we obtain that expression (got it through Claude Haiku 4.5)
        return (np.sum(loss) / sample_count, dLdx)


def softmax(x: Array) -> Array:
    """
    Arguments:
        x: Matrice des entrées. dim(N, I).
    Returns:
        out: Matrice des valeurs projetées. dim(N, I).
    """
    return (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)).astype(x.dtype)


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
        loss = np.sum((x - target) ** 2, dtype=x.dtype, axis=0) / x.shape[1]
        dLdx = (2 * (x - target)).astype(x.dtype) / x.shape[1]
        return (np.sum(loss) / sample_count, dLdx / sample_count)
