# Python imports

# Self imports
from Optimization.Optimizers import Sgd
# Other imports

class NeuralNetwork:
    def __init__(self, optimizer: Sgd) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        