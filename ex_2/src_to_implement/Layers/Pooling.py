# Python imports
from typing import Union

# Third party imports
import numpy as np

# Self imports


class Pooling:
    def __init__(self, stride_shape: Union[tuple, int], pooling_shape: int) -> None:
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        pass

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass
