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
        ## Weight_tensor's number of rows must be equals to the number of rows of the next layers input_tensor which is 
        ## the output_size in this case.
        self.weights = np.tile(np.random.uniform(0,1, input_size), (output_size, 1))

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        Gets a input_tensor with shape(number_of_samples, pixels).
        Params -> input_tensor: np.ndarray shape(ouput_size, number_of_samples)
        '''
        wx = self.weights.dot(input_tensor)
        wx_shape = wx.shape
        bais = np.tile(np.random.uniform(0,1, wx_shape[1]), (wx_shape[0], 1))
        wx_plus_b = wx+bais
        self.output_tensor = wx_plus_b
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