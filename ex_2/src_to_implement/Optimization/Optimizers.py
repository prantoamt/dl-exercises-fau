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


class Sgd(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        return weight_tensor - (self.learning_rate * gradient_tensor)


class SgdWithMomentum(Optimizer):
    """
    Stochastic gradient decent with momentum.
    """

    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = 0  ## The initial momentum is ZERO

    def calculate_update(
        self, weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        @ Params:
            weight_tenson: old weight tensor -> np.ndarray
            gradient_tensor: gradient with respect to the old weight tensor -> np.ndarray

        v(k) = momentum_rate * v(k-1) + learning_rate * gradient(k).
        w(k+1) = w(k) - v(k)
        v(k) denotes to the new momentum at k's iteration.
        w(k) denotes to new weights after considering momentum at k's iteration.
        gradient(k) denotes to the gradient of k's iteration.
        Assume the very first momentum v(0) is ZERO
        """
        self.v_k = self.momentum_rate * self.v_k + (
            self.learning_rate * gradient_tensor
        )
        w_k_plus_one = weight_tenson - self.v_k
        return w_k_plus_one


class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.iter = 1

    def calculate_update(
        self, weight_tenson: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        @ Params:
            weight_tenson: old weight tensor -> np.ndarray
            gradient_tensor: gradient with respect to the old weight tensor -> np.ndarray

        v(k) = mu * v(k-1) + (1 - mu) * gradient(k)
        r(k) = rho * r(k-1) + (1 - rho) * (gradient(k))^2
        v(k)hat = v(k) / (1-mu^k)
        r(k)hat = r(k) / (1-rho^k)
        w(k+1) = w(k) - learning_rate * (v(k)hat / sqrt(r(k)hat)+eps)

        v(k) denotes to the new gradient with momentum at k's iteration.
        w(k) denotes to new weights after considering momentum at k's iteration.
        gradient(k) denotes to the gradient of k's iteration.
        Assume the very first momentum v(0) is ZERO and r(0) = 0
        """
        # print(
        #     f"v({self.iter-1}) = {self.v_k}, r({self.iter-1}) = {self.r_k}, w({self.iter}) = {weight_tenson}"
        # )
        self.v_k = self.mu * self.v_k + ((1 - self.mu) * gradient_tensor)
        self.r_k = self.rho * self.r_k + ((1 - self.rho) * gradient_tensor**2)
        v_k_hat = self.v_k / (1.0 - self.mu**self.iter)
        r_k_hat = self.r_k / (1.0 - self.rho**self.iter)
        w_k_plus_one = weight_tenson - (
            self.learning_rate
            * (v_k_hat / (np.sqrt(r_k_hat) + np.finfo(np.float64).eps))
        )
        # print(
        #     f"v({self.iter}) = {self.v_k}, r({self.iter}) = {self.r_k}, v({self.iter})^ = {v_k_hat}, r({self.iter})^ = {r_k_hat}, w({self.iter}+1) = {w_k_plus_one}"
        # )
        self.iter += 1
        return w_k_plus_one
