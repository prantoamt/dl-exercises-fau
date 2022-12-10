# Python imports
from typing import Union

# Third party imports
import numpy as np

# Self imports
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.Initializers import Initializer
from Layers.SoftMax import SoftMax
from Optimization.Optimizers import Optimizer


class NeuralNetwork:
    def __init__(
        self,
        optimizer: Optimizer,
        weight_initializer: Initializer,
        bias_initializer: Initializer,
    ) -> None:
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers: FullyConnected = []
        self.data_layer = None  # Contains tuple (input tensor, label tensor)
        self.loss_layer = None
        self.input_tensor = None  # To store input tensor
        self.label_tensor = None  # To store label tensor

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
                weight_initializer=self.weight_initializer,
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
