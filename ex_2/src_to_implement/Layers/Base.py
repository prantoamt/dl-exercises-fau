# Python imports
import copy

# Third party imports
import numpy as np

# Self imports
from Optimization.Optimizers import Optimizer


class BaseLayer:
    def __init__(self) -> None:
        self.trainable = False
        self.weights = None
        self._gradient_weights = None
        self._optimizer = None

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        self._optimizer = copy.deepcopy(value)

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad_weights: np.ndarray) -> None:
        self._gradient_weights = grad_weights
