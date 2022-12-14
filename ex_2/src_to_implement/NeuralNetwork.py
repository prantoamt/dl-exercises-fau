# Python imports
from typing import Union, Tuple

# Third party imports
import numpy as np

# Self imports
from Optimization.Loss import CrossEntropyLoss
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.Initializers import Initializer
from Layers.SoftMax import SoftMax
from Optimization.Optimizers import Optimizer


class NeuralNetwork:
    def __init__(
        self,
        optimizer: Optimizer,
        weights_initializer: Initializer,
        bias_initializer: Initializer,
    ) -> None:
        self.optimizer: Optimizer = optimizer
        self.weights_initializer: Initializer = weights_initializer
        self.bias_initializer: Initializer = bias_initializer
        self.loss: list[int] = []
        self.layers: list[BaseLayer] = []
        self.data_layer: Tuple[int] = None
        self.loss_layer: CrossEntropyLoss = None
        self.input_tensor: np.ndarray = None
        self.label_tensor: np.ndarray = None

    def forward(self) -> np.ndarray:
        # Fetch the input_tensor, label_tensor from data_layer
        self.input_tensor, self.label_tensor = self.data_layer.next()

        # Pass the data tensor in each layer of the network
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        # Return last layer's (loss layer) output, which is technically the Cross Entropy Loss
        loss_output = self.loss_layer.forward(self.input_tensor, self.label_tensor)

        return loss_output

    def backward(self) -> np.ndarray:
        # Propagate backward by starting from loss layer
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Pass the error_tensor found from loss_layer in each layer of the network reversely
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer: Union[FullyConnected, ReLU, SoftMax]) -> None:
        # If the layer is trainable, set the network's optimizer as layer's optimizer
        if layer.trainable:
            layer.optimizer = self.optimizer
            layer.initialize(
                weights_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
            )

        self.layers.append(layer)

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            loss_output = self.forward()  # Forward Propagation
            self.loss.append(loss_output)  # Store the loss
            self.backward()  # Backward Propagation

    def test(self, input_tensor: np.ndarray) -> None:
        output_tensor = input_tensor

        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)  # Forward Propagation

        return output_tensor
