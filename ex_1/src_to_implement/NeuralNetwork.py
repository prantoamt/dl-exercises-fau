# Python imports

# Self imports
from Optimization.Optimizers import Sgd
# Other imports
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer: Sgd) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass

    def append_layer(self, layer) -> None:
        pass

    def train(self, iterations: int) -> None:
        pass

    def test(self, input_tensor: np.ndarray) -> None:
        pass