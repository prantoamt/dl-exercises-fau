# Python imports
from abc import ABC, abstractmethod
from typing import Tuple, int, Optional

# Third party imports
import numpy as np

# Self imports


class Initializer(ABC):
    """
    Base abstruct class that will force the subsclasses to define the abstruct methods
    """

    @abstractmethod
    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
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

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        return super().initializer(weights_shape, fan_in, fan_out)


class UniformRandom(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to
        uniform destribution from [0,1)
         @Params:
            weight_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        """
        pass


class Xavier(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to
        the uniform destribution range created by Xaviar/Glorot equation.
        @Params:
            weight_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        Explaination:
            sigma = sqrt( 2 / fan_out + fan_in )
            The weight_tensor will have random numbers that belongs to the
            uniform destribution from [0, sigma).
        """
        pass


class He(Initializer):
    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to
        the uniform destribution range created by He equation.
        @Params:
            weight_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        Explaination:
            sigma = sqrt( 2 / fan_in )
            The weight_tensor will have random numbers that belongs to the
            uniform destribution from [0, sigma).
        """
        pass
