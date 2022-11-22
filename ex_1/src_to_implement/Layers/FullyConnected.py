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
        self.optimizer = None
        ## Fully connected layers must be trainable.
        self.trainable = True
        ## Weight_tensor's number of columns must be equals to the number of columns of the next layers input_tensor which is 
        ## the output_size in this case. And the row size is the input_size+1.
        ## The last one row is the biases.
        self.weights = np.tile(np.random.uniform(0,1, output_size), (input_size+1, 1))

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        Gets a input_tensor with shape(number_of_samples, pixels).
        Params -> input_tensor: np.ndarray shape(number_of_samples, pixels)
        '''
        bias_unit = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.append(input_tensor, bias_unit, axis=1)
        self.output_tensor = input_tensor.dot(self.weights)
        # Return the output tensor. This will be the input_tensor forn the next layer.
        return self.output_tensor

    @property
    def optimizer(self) -> Sgd:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Sgd) -> None:
        self._optimizer = copy.deepcopy(value)

    # def _one_hot(self, output: np.array) -> np.array:
    #     one_hot_output = np.zeros((output.size, output.max()+1))
    #     one_hot_output[np.arange(output.shape, output)] = 1
    #     one_hot_output = one_hot_output.T
    #     return one_hot_output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass