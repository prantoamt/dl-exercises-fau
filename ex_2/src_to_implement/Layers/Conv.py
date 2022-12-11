# Python imports
from typing import Union

# Third party imports
import numpy as np

# Self imports
from Layers.Initializers import Initializer
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(
        self,
        stride_shape: Union[tuple, int],
        convolution_shape: tuple,
        num_kernels: int,
    ) -> None:
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self._gradient_bais = None

    def initialize(
        weight_initializer: Initializer, bias_initializer: Initializer
    ) -> tuple:
        '''
        Initializes weight and bias tensor with the given weight
        and bias initializer objects.
        @Params:
            weight_initializer -> Initializer
            bias_initializer -> Initializer
        """
        '''
        pass

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        pass

    @property
    def gradient_bais(self) -> np.ndarray:
        return self._gradient_bais

    @gradient_bais.setter
    def gradient_bais(self, grad_bais: np.ndarray) -> None:
        self._gradient_bais = grad_bais
