# Python imports
import copy
from abc import ABC, abstractmethod

# Third party imports

# Self imports
from Layers.Initializers import Initializer
from Optimization.Optimizers import Optimizer


class BaseLayer(ABC):
    """
    Base abstruct layer that will force abstruct methods to define.
    """

    def __init__(self) -> None:
        self.trainable = False
        self.weights = None
        self._gradient_weights = None
        self._optimizer = None

    @abstractmethod
    def initialize(
        weight_initializer: Initializer, bias_initializer: Initializer
    ) -> tuple:
        pass

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
