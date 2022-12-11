# Python imports

# Third party imports
import numpy as np

# Self imports
from Layers.Base import BaseLayer
from Layers.Initializers import Initializer


class Flatten(BaseLayer):
    def __init__(self) -> None:
        self.trainable = False

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        @Params:
            input_tensor -> np.ndarray
        returns:
            flatten_tensor-> np.ndarray
        Flattens the input tenson to be compatible with fully connected layes.
        The input tensor in this layer is expected to be 4D.
        For example: (9,3,4,11),
            where, 9 is batch size. 3 is the chanel number of an image
            4 is heigh and 11 is width of an image.
        Keeping the batch size same, the function flatten the 4D array and
        returns the result.
        """
        self.input_tensor = input_tensor
        return input_tensor.reshape(len(input_tensor), -1)

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        @Params:
            error_tensor(flattened) -> np.ndarray
        returns:
            un_flatten_tensor-> np.ndarray
        Error is the a flatten tensor comming back from densed or fully connected
        layers. The function reshapes the error tensor to the previous shape it
        was to be compatible with convolution layers.
        """
        return error_tensor.reshape(self.input_tensor.shape)

    def initialize(
        weight_initializer: Initializer, bias_initializer: Initializer
    ) -> tuple:
        return super().initialize(bias_initializer)
