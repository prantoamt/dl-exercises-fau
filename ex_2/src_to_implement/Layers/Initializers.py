# Python imports
from abc import ABC, abstractmethod
from typing import Tuple, int, Optional

# Third party imports
import numpy as np

# Self imports


class Initializer(ABC):
    """
    Abstruct class that will force the subsclasses to override the abstruct methods
    """

    @abstractmethod
    def initializer(
        self, weights_shape: Tuple, fan_in: int, fan_out: int
    ) -> np.ndarray:
        """
        Returns the initialized tensor of desired size.
        """
        pass


class Constant(Initializer):
    """
    For initializing Biases
    """

    def __init__(self, constant: Optional[float] = 0.1) -> None:
        super().__init__()
        self.constant = constant

    def initializer(
        self, weights_shape: Tuple, fan_in: int, fan_out: int
    ) -> np.ndarray:
        return super().initializer(weights_shape, fan_in, fan_out)


class UniformRandom(Initializer):
    def initializer(
        self, weights_shape: Tuple, fan_in: int, fan_out: int
    ) -> np.ndarray:
        return super().initializer(weights_shape, fan_in, fan_out)


class Xavier(Initializer):
    def initializer(
        self, weights_shape: Tuple, fan_in: int, fan_out: int
    ) -> np.ndarray:
        return super().initializer(weights_shape, fan_in, fan_out)


class He(Initializer):
    def initializer(
        self, weights_shape: Tuple, fan_in: int, fan_out: int
    ) -> np.ndarray:
        return super().initializer(weights_shape, fan_in, fan_out)
