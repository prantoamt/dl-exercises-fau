# Python imports

# Self imports
from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd

# Other imports
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        super().trainable = True
        self.weights = np.random.uniform(0,1, output_size)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.output_tensor = self.weights.dot(input_tensor)
        return self.output_tensor

    @property
    def optimizer(self) -> Sgd:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Sgd) -> None:
        self._optimizer = value

    # def _one_hot(self, output: np.array) -> np.array:
    #     one_hot_output = np.zeros((output.size, output.max()+1))
    #     one_hot_output[np.arange(output.shape, output)] = 1
    #     one_hot_output = one_hot_output.T
    #     return one_hot_output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass