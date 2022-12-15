# Python imports
from typing import Union, Tuple

# Third party imports
import numpy as np
from scipy import signal

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
            self.weights = np.random.rand(
                *(num_kernels, convolution_shape[0], convolution_shape[1])
            )
            self.bias = UniformRandom().initialize(
                (1, self.num_kernels), 1, self.num_kernels
            )
        elif len(convolution_shape) == 3:
            self.weights = np.random.rand(
                *(
                    num_kernels,
                    convolution_shape[0],
                    convolution_shape[1],
                    convolution_shape[2],
                )
            )
            self.bias = np.random.rand(num_kernels)

    def initialize(
        self, weights_initializer: Initializer, bias_initializer: Initializer
    ) -> Tuple:
        """
        Initializes weight and bias tensor with the given weight
        and bias initializer objects.
        @Params:
            weight_initializer -> Initializer
            bias_initializer -> Initializer
        """
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize(
                (
                    self.num_kernels,
                    self.convolution_shape[0],
                    self.convolution_shape[1],
                    self.convolution_shape[2],
                ),
                self.convolution_shape[0]
                * self.convolution_shape[1]
                * self.convolution_shape[2],
                self.num_kernels
                * self.convolution_shape[1]
                * self.convolution_shape[2],
            )
            self.bias = bias_initializer.initialize(
                (self.num_kernels), 1, self.num_kernels
            )

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize(
                (
                    self.num_kernels,
                    self.convolution_shape[0],
                    self.convolution_shape[1],
                ),
                self.convolution_shape[0] * self.convolution_shape[1],
                self.num_kernels * self.convolution_shape[1],
            )
            self.bias = bias_initializer.initialize(
                (1, self.num_kernels), 1, self.num_kernels
            )

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor

        if len(self.convolution_shape) == 2:
            self.batch_size = self.input_tensor.shape[0]
            self.input_height = self.input_tensor.shape[1]
            self.input_width = self.input_tensor.shape[2]
            self.output_height = int(np.ceil(self.input_height / self.stride_shape[0]))
            self.output_width = int(np.ceil(self.input_width / self.stride_shape[1]))
            self.output_shape = (
                self.batch_size,
                self.num_kernels,
                self.output_height,
                self.output_width,
            )
        elif len(self.convolution_shape) == 3:
            self.batch_size = self.input_tensor.shape[0]
            self.input_channel = self.input_tensor.shape[1]
            self.input_height = self.input_tensor.shape[2]
            self.input_width = self.input_tensor.shape[3]
            self.output_height = int(np.ceil(self.input_height / self.stride_shape[0]))
            self.output_width = int(np.ceil(self.input_width / self.stride_shape[1]))
            self.output_shape = (
                self.batch_size,
                self.num_kernels,
                self.output_height,
                self.output_width,
            )
        self.output_tensor = np.zeros(self.output_shape)

        ## Algorithm:
        ## Take one image/data
        ## Take one kernel
        ## Take one channel of the kernel
        ## Perform cross-corelation on each channel of the image with each channel of the kernel
        ## sum all the channel's cross-corelation result along 0 dimention.

        for item in range(self.batch_size):  ## Take one image/data
            for kernel in range(self.weights.shape[0]):  ## Take one kernel
                channel_corelation = []
                for channel in range(self.weights.shape[1]):  ## Take Channel
                    channel_corelation.append(
                        ## Perform cross-corelation on each channel of the image with each channel of the kernel
                        signal.correlate(
                            self.input_tensor[item, channel],
                            self.weights[kernel, channel],
                            mode="same",
                            method="direct",
                        )
                    )
                channel_corelation = np.array(channel_corelation)
                ## sum all the channel's cross-corelation result along 0 dimention.
                channel_corelation = channel_corelation.sum(axis=0)

                if len(self.convolution_shape) == 3:
                    channel_corelation = channel_corelation[
                        :: self.stride_shape[0], :: self.stride_shape[1]
                    ]
                elif len(self.convolution_shape) == 2:
                    channel_corelation = channel_corelation[:: self.stride_shape[0]]

                self.output_tensor[item, kernel] = (
                    channel_corelation + self.bias[kernel]
                )

        return self.output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass

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
