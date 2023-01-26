# Python imports
from typing import Union, Tuple
import copy

# Third party imports
import numpy as np
from scipy import signal, ndimage

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
        super().__init__()

        self.stride_shape: Union[Tuple, int] = stride_shape
        self.convolution_shape: Tuple = convolution_shape
        self.num_kernels: int = num_kernels
        self.trainable: bool = True
        self._gradient_weights: np.ndarray = None
        self._gradient_bias: np.ndarray = None
        self._optimizer = None

        if len(convolution_shape) == 2:
            self.weights = np.random.rand(
                num_kernels, convolution_shape[0], convolution_shape[1]
            )
        elif len(convolution_shape) == 3:
            self.weights = np.random.rand(
                num_kernels,
                convolution_shape[0],
                convolution_shape[1],
                convolution_shape[2],
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
                (self.num_kernels), self.num_kernels, 1
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
                (1, self.num_kernels), self.num_kernels, 1
            )

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor

        if len(self.convolution_shape) == 2:
            self.batch_size = self.input_tensor.shape[0]
            self.input_height = self.input_tensor.shape[1]
            self.input_width = self.input_tensor.shape[2]
            self.output_width = int(np.ceil(self.input_width / self.stride_shape[0]))
            self.output_shape = (
                self.batch_size,
                self.num_kernels,
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
        # Rearrange Weight Metrics from (K, C, H, W) to (C, K, H, W)
        new_weights = []
        for channel in range(self.convolution_shape[0]):
            all_kernels = []
            for kernel in range(self.num_kernels):
                all_kernels.append(self.weights[kernel, channel])
            new_weights.append(all_kernels)
        new_weights = np.array(new_weights)

        # Up sampling error tensor to fixed the stride downsampling
        up_sampled_err_arr = []
        for batch_ind in range(error_tensor.shape[0]):
            err_img = error_tensor[batch_ind]
            new_channel_arr = []
            for err_channel in range(error_tensor.shape[1]):
                if len(self.convolution_shape) == 3:
                    new_zero_arr = np.zeros((self.input_height, self.input_width))

                    for err_row in range(err_img.shape[-2]):
                        for err_col in range(err_img.shape[-1]):
                            new_zero_arr[
                                err_row * self.stride_shape[0],
                                err_col * self.stride_shape[1],
                            ] = err_img[err_channel, err_row, err_col]
                    new_channel_arr.append(new_zero_arr)

                elif len(self.convolution_shape) == 2:
                    new_zero_arr = np.zeros((self.input_width))

                    for err_col in range(err_img.shape[-1]):
                        new_zero_arr[err_col * self.stride_shape[0]] = err_img[
                            err_channel, err_col
                        ]
                    new_channel_arr.append(new_zero_arr)

            up_sampled_err_arr.append(new_channel_arr)
        up_sampled_err_arr = np.array(up_sampled_err_arr)

        # Calculating Previous error tensor E(n-1)
        previous_err_tensor = []
        for batch_ind in range(error_tensor.shape[0]):
            error_channel = []
            for channel in range(self.convolution_shape[0]):
                convolve_err = signal.convolve(
                    up_sampled_err_arr[batch_ind],
                    np.flip(new_weights[channel], 0),
                    mode="same",
                )
                if self.num_kernels <= 2:
                    error_channel.append(convolve_err[self.num_kernels - 1])
                else:
                    error_channel.append(convolve_err[self.num_kernels // 2])
            previous_err_tensor.append(error_channel)
        previous_err_tensor = np.array(previous_err_tensor)

        # Gradient calculation
        self.gradient_weights = np.zeros((self.weights.shape))
        self.gradient_bias = np.zeros((self.bias.shape))

        for batch_ind in range(error_tensor.shape[0]):
            err_img = error_tensor[batch_ind]
            new_kernel_arr = []
            error_bias = []
            for kernel in range(self.num_kernels):
                new_channel_arr = []
                for channel in range(self.convolution_shape[0]):
                    if len(self.convolution_shape) == 3:
                        padding_height = self.convolution_shape[1] // 2
                        padding_width = self.convolution_shape[2] // 2
                        padded_input = np.pad(
                            self.input_tensor[batch_ind, channel],
                            (
                                (padding_height, padding_height),
                                (padding_width, padding_width),
                            ),
                        )
                        output = signal.correlate2d(
                            padded_input,
                            up_sampled_err_arr[batch_ind][kernel],
                            mode="valid",
                        )
                        output = output[
                            : self.weights.shape[2], : self.weights.shape[3]
                        ]
                        new_channel_arr.append(output)
                    elif len(self.convolution_shape) == 2:
                        channel_array = self.input_tensor[batch_ind, channel]
                        if self.stride_shape[0] == 1:
                            output = ndimage.correlate1d(
                                channel_array, weights=err_img[kernel]
                            )
                        else:
                            output = signal.correlate2d(
                                channel_array.reshape(-1, 1),
                                up_sampled_err_arr[batch_ind][kernel].reshape(-1, 1),
                                mode="valid",
                            )[:, 0]
                        new_channel_arr.append(output)

                new_kernel_arr.append(new_channel_arr)
                error_bias.append(np.sum(err_img[kernel]))

            error_bias = np.array(error_bias)
            new_kernel_arr = np.array(new_kernel_arr)
            self.gradient_bias = np.add(self.gradient_bias, error_bias)
            self.gradient_weights = np.add(self.gradient_weights, new_kernel_arr)

        # Update weight and bias
        if self.optimizer:
            self.weights = self.weights_optimizer.calculate_update(
                self.weights, self.gradient_weights
            )
            self.bias = self.bias_optimizer.calculate_update(
                self.bias, self.gradient_bias
            )

        return previous_err_tensor

    @property
    def gradient_bias(self) -> np.ndarray:
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, grad_bais: np.ndarray) -> None:
        self._gradient_bias = grad_bais

    @property
    def gradient_weight(self) -> np.ndarray:
        return self._gradient_weight

    @gradient_weight.setter
    def gradient_weight(self, grad_weight: np.ndarray) -> None:
        self._gradient_weight = grad_weight

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer = copy.deepcopy(self._optimizer)
