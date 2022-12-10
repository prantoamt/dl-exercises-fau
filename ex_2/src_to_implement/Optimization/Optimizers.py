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

    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

    def calculate_update(
        weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        @ Params:
            weight_tenson: old weight tensor -> np.ndarray
            gradient_tensor: gradient with respect to the old weight tensor -> np.ndarray

        v(K) = momentum_rate * v(k-1) - learning_rate * gradient(K).
        w(K+1) = w(K) + v(K)
        v(K) denotes to the new gradient with momentum at k's iteration.
        w(K) denotes to new weights after considering momentum at k's iteration.
        gradient(K) denotes to the gradient of k's iteration.
        """
        pass


class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

    def calculate_update(
        weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        @ Params:
            weight_tenson: old weight tensor -> np.ndarray
            gradient_tensor: gradient with respect to the old weight tensor -> np.ndarray

        v(K) = mu * v(k-1) + (1 - mu) * gradient(K)
        r(K) = rho * r(k-1) + (1 - rho) * (gradient(K))^2
        v(K)hat = v(K) / (1-mu)
        r(K)hat = r(K) / (1-rho)
        w(K+1) = w(K) - learning_rate * (v(K)hat / sqrt(r(k)hat)) + eps

        v(K) denotes to the new gradient with momentum at k's iteration.
        w(K) denotes to new weights after considering momentum at k's iteration.
        gradient(K) denotes to the gradient of k's iteration.
        """
        pass
