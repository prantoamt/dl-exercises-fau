# Python imports
from typing import List, Union

# Self imports
from Optimization.Optimizers import Sgd
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Layers.Helpers import IrisData

# Other imports
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer: Sgd) -> None:
        self.optimizer: Sgd = optimizer
        self.loss = []
        self.layers: List[FullyConnected] = []
        self.data_layer: IrisData = None
        self.loss_layer = None

    def forward(self) -> np.ndarray:
        '''
        Fetch a tuple (data, label) from data_layer.next(),
        where each row represents an image and each column represents pixel.
        Params ->  None
        Returns -> last_layer_output_tensor: np.ndarray
        '''
        data_tensor, label_tensor = self.data_layer.next()
        ## Pass the data tensor in each layer of the network.
        last_layer_output_tensor = None
        for layer in self.layers:
            last_layer_output_tensor = layer.forward(data_tensor)

        # Return last layer's output tensor. which is technically the predictions.
        return last_layer_output_tensor

    def backward(self) -> np.ndarray:
        pass

    def append_layer(self, layer: Union[FullyConnected, ReLU, SoftMax]) -> None:
        # If the layer is trainable, sett the network's optimizer as layer's optimizer.
        # Append the layers
        if layer.trainable:
            layer.optimizer = self.optimizer
        self.layers.append(layer)

    def train(self, iterations: int) -> None:
        pass

    def test(self, input_tensor: np.ndarray) -> None:
        pass