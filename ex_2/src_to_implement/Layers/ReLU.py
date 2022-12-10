# Python imports

# Self imports
from Layers.Base import BaseLayer

# Other imports
import numpy as np


class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = None  # To store the output of ReLU, which will be needed to compute backward propagation later

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        relu_output = np.maximum(
            0, input_tensor
        )  # calculate output of the ReLU Activation Function
        self.cache = np.copy(
            relu_output
        )  # stored the output of ReLU for backward propagation

        return relu_output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        relu_derivative = np.where(self.cache <= 0, 0, 1)
        gradient = (
            error_tensor * relu_derivative
        )  # gradient of the cost with respect to output of ReLU

        return gradient
