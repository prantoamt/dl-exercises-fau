# Python imports
from abc import ABC, abstractmethod
from typing import Tuple, Optional

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
    def __init__(self, constant: Optional[float] = 0.1) -> None:
        super().__init__()
        self.constant = constant

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with constant number
         @Params:
            weights_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        """
        return np.full(weights_shape, self.constant)


class UniformRandom(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to uniform destribution from [0,1)
         @Params:
            weights_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        """
        # return np.random.uniform(0, 1, fan_in*fan_out).reshape(weights_shape)
        return np.random.rand(fan_in, fan_out)


class Xavier(Initializer):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to
        the uniform destribution range created by Xaviar/Glorot equation.
        @Params:
            weights_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        Explaination:
            sigma = sqrt(2 / fan_out + fan_in)
            The weight_tensor will have random numbers that belongs to the
            normal destribution from [0, sigma).
        """
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, sigma, weights_shape)


class He(Initializer):
    def __init__(self) -> None:
        super().__init__()
    
    def initialize(self, weights_shape: Tuple, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initializes weights with random numbers that belongs to
        the uniform destribution range created by He equation.
        @Params:
            weights_shape: the desired shape of to be initialized weight_tensor
            fan_in: input size of the layer -> int
            fan_out: output size of the layer -> int
        Explaination:
            sigma = sqrt(2 / fan_in)
            The weight_tensor will have random numbers that belongs to the
            normal destribution from [0, sigma).
        """
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0.0, sigma, weights_shape)
