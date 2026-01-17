from dnn_framework.optimizer import Optimizer
from dnn_framework.common import Array


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate: float = 0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def _step_parameter(
        self, parameter: Array, parameter_grad: Array, parameter_name: str
    ) -> Array:
        if parameter.shape != parameter_grad.shape:
            parameter_grad = parameter_grad.T
        parameter -= self.learning_rate * parameter_grad
        self._parameters[parameter_name] = parameter
        return parameter
