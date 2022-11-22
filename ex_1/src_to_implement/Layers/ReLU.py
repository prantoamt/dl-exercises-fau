# Python imports

# Self imports
from Layers.Base import BaseLayer

# Other imports
import numpy as np

class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        pass

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        pass