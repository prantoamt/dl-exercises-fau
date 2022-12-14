# Python imports
from typing import Union, Tuple

# Third party imports
import numpy as np

# Self imports
from Layers.Initializers import Initializer, UniformRandom
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(
        self,
        stride_shape: Union[Tuple, int],
        convolution_shape: Tuple,
        num_kernels: int,
    ) -> None:
        self.stride_shape: Union[Tuple, int] = stride_shape
        self.convolution_shape: Tuple = convolution_shape
        self.num_kernels: int = num_kernels
        self.trainable: bool = True
        self._gradient_weights: np.ndarray = None
        self._gradient_bais: np.ndarray = None

        if len(convolution_shape) == 2:
            self.kernels = np.random.rand(
                *(num_kernels, convolution_shape[0], convolution_shape[1])
            )
        elif len(convolution_shape) == 3:
            self.kernels = np.random.rand(
                *(
                    num_kernels,
                    convolution_shape[0],
                    convolution_shape[1],
                    convolution_shape[2],
                )
            )

    def initialize(
        self, weight_initializer: Initializer, bias_initializer: Initializer
    ) -> Tuple:
        """
        Initializes weight and bias tensor with the given weight
        and bias initializer objects.
        @Params:
            weight_initializer -> Initializer
            bias_initializer -> Initializer
        """
        self.weights = weight_initializer.initialize(
            self.convolution_shape,
            np.prod(self.convolution_shape),
            np.prod(self.convolution_shape[1:]) * self.num_kernels,
        )
        self.bais = bias_initializer.initialize(
            self.convolution_shape,
            np.prod(self.convolution_shape),
            np.prod(self.convolution_shape[1:]) * self.num_kernels,
        )

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor

        if len(self.convolution_shape) == 2:
            output_shape = (
                self.input_tensor.shape[0],
                self.num_kernels,
                int(np.ceil(self.input_tensor.shape[1] / self.stride_shape[0])),
                int(np.ceil(self.input_tensor.shape[2] / self.stride_shape[0])),
            )
        elif len(self.convolution_shape) == 3:
            output_shape = (
                self.input_tensor.shape[0],
                self.num_kernels,
                int(np.ceil(self.input_tensor.shape[2] / self.stride_shape[0])),
                int(np.ceil(self.input_tensor.shape[3] / self.stride_shape[0])),
            )

        output_tensor = np.zeros(output_shape)
        print(output_shape, "=================")
        return output_tensor

    @property
    def gradient_bais(self) -> np.ndarray:
        return self._gradient_bais

    @gradient_bais.setter
    def gradient_bais(self, grad_bais: np.ndarray) -> None:
        self._gradient_bais = grad_bais

    @property
    def gradient_weight(self) -> np.ndarray:
        return self._gradient_weight

    @gradient_weight.setter
    def gradient_weight(self, grad_weight: np.ndarray) -> None:
        self._gradient_weight = grad_weight
