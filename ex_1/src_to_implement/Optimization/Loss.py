# Python imports

# Self imports

# Other imports
import numpy as np


class CrossEntropyLoss:
    def __init__(self) -> None:
        self.cache = None  # To store prediction_tensor for backward propagation

    def forward(
        self, prediction_tensor: np.ndarray, label_tensor: np.ndarray
    ) -> np.ndarray:
        self.cache = np.copy(
            prediction_tensor
        )  # store prediction_tensor for backward propagation

        pred_tensor_ones = prediction_tensor[
            label_tensor == 1
        ]  # find the predicted value which has label 1
        cross_entropy_loss = np.sum(
            -1 * np.log(pred_tensor_ones + np.finfo(pred_tensor_ones.dtype).eps)
        )  # calculate the cross entropy loss

        return cross_entropy_loss

    def backward(self, label_tensor: np.ndarray) -> np.ndarray:
        error_tensor = -1 * (
            label_tensor / self.cache
        )  # calculate the error tensor using the backward equation of Cross Entropy Loss

        return error_tensor
