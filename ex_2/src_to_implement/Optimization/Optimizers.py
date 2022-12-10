# Python imports
from abc import ABC, abstractmethod

# Third party imports
import numpy as np

# Self imports


class Optimizer(ABC):
    """
    Base abstruct class for Optimizers that will force subclasses to
    define abstruct methods.
    """

    @abstractmethod
    def calculate_update(
        weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        pass


class SgdWithMomentum(Optimizer):
    """
    Stochastic gradient decent with momentum.
    """

    def calculate_update(
        weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        @ Params:
            weight_tenson: old wight tensor -> np.ndarray
            gradient_tensor: gradient with respect to the old weight tensor -> np.ndarray

        v(K) = momentum_rate * v(k-1) - learning_rate * gradient.
        w(K+1) = w(K) + v(K)
        v(K) denotes to the new gradient with momentum at k's iteration.
        w(K) denotes to new weights after considering momentum at k's iteration.
        """

        pass
