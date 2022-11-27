# Python imports

# Self imports
from Layers.Base import BaseLayer

# Other imports
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.cache = None  # To store the output of SoftMax (class probabilities), which will be needed to compute backward propagation later

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        input_tensor_prime = input_tensor - np.max(input_tensor)  # increase numerical stability input_tensor (x) is shifted, x'=x-max(x)
        exp_input_tensor_prime = np.exp(input_tensor_prime)
        sum_exp_each_class = np.sum(exp_input_tensor_prime, axis=1, keepdims=True)
        softmax_output = (exp_input_tensor_prime / sum_exp_each_class)  # calculate output of the SoftMax Activation Function
        
        self.cache = np.copy(softmax_output)  # stored the output of SoftMax for backward propagation

        return softmax_output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # implemented the backward propagation equation of SoftMax (1_BasicFramework.pdf, pg 16)
        sum_err_and_class_prob = np.sum(np.multiply(error_tensor, self.cache), axis=1, keepdims=True)
        gradient = np.multiply(self.cache, error_tensor - sum_err_and_class_prob)

        return gradient
