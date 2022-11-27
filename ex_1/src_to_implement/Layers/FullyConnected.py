# Python imports
import copy

# Self imports
from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd

# Other imports
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.trainable = True  # Fully connected layers must be trainable
        self._optimizer = None
        self.input_size = input_size
        self.output_size = output_size

        self.weight_init = np.random.rand(
            input_size, output_size
        )  # weight = (in_size, out_size)
        self.bias = np.random.rand(1, output_size)  # bias = (1, out_size)
        self.weights = np.concatenate(
            [self.weight_init, self.bias]
        )  # final_weights = (in_size+1, out_size) [merged in row]

        self.input_tensor = None
        self._gradient_weights = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        batch_sz, input_sz = input_tensor.shape  # x = (batch_size, in_size)
        input_tensor_0 = np.ones((batch_sz, 1))  # x0 = (batch_size, 1)
        input_tensor_merged = np.concatenate(
            [input_tensor, input_tensor_0], axis=1
        )  # x_final = (batch_size, in_size+1) [merged in col]
        output_tensor = np.dot(
            input_tensor_merged, self.weights
        )  # output_tensor = x_final * final_weights = (batch_size, out_size)

        self.input_tensor = input_tensor_merged

        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        new_error_tensor = np.dot(
            error_tensor, self.weights.T
        )  # calculate the error tensor to pass in the previous layer
        self._gradient_weights = np.dot(
            self.input_tensor.T, error_tensor
        )  # calculate gradient to update the weights using optimizer

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(
                self.weights, self._gradient_weights
            )  # update the weights

        return new_error_tensor[
            :, : self.input_size
        ]  # to match the previous layers output size, slice the col size accordingly

    @property
    def optimizer(self) -> Sgd:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Sgd) -> None:
        self._optimizer = value

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad_weights: np.ndarray) -> None:
        self._gradient_weights = grad_weights
