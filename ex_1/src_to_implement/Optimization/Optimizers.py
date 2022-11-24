# Python imports

# Self imports

# Other imports
import numpy as np


class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        return weight_tensor - (self.learning_rate * gradient_tensor)
