# Python imports
from abc import ABC, abstractclassmethod
from typing import Tuple

# Self imports

# Other imports
import numpy as np
import matplotlib.pyplot as plt


class Pattern(ABC):
    """
    An abstruct class for patterns
    """

    def __init__(self) -> None:
        super().__init__()
        self.output = None

    @abstractclassmethod
    def draw(self) -> None:
        """
        creates the pattern using numpy functions
        """
        pass

    def show(self) -> None:
        """
        Visualizes the pattern
        """
        fig, ax = plt.subplots()
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Circle(Pattern):
    def __init__(
        self, 
        resolution: int, 
        radius: int, 
        position: Tuple[int, int]
        ) -> None:
        super().__init__()
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self) -> np.ndarray:
        xx, yy = np.mgrid[: self.resolution, : self.resolution]
        circle = (xx - self.position[1]) ** 2 + (
            yy - self.position[0]
        ) ** 2 <= self.radius**2
        self.output = circle.astype(int)
        return np.copy(self.output)


class Checker(Pattern):
    def __init__(self, resolution: int, tile_size: int) -> None:
        super().__init__()
        valid_resolution = True if resolution % (tile_size * 2) == 0 else False
        if valid_resolution:
            self.resolution = resolution
            self.tile_size = tile_size
        else:
            raise ValueError(
                "Please enter the inputs such that resolution is divible by 2*tile_size"
            )

    def draw(self) -> np.ndarray:
        total_boxes_for_one_row_in_pair = (
            int((self.resolution / self.tile_size) / 2)
            )
        black_box = np.zeros((self.tile_size, self.tile_size), dtype=int)
        white_box = np.ones((self.tile_size, self.tile_size), dtype=int)
        b_w_box_in_pair = np.concatenate((black_box, white_box), axis=1)
        w_b_box_in_pair = np.concatenate((white_box, black_box), axis=1)

        first_row = np.tile(b_w_box_in_pair, total_boxes_for_one_row_in_pair)
        second_row = np.tile(w_b_box_in_pair, total_boxes_for_one_row_in_pair)
        comb_first_second_rows = np.concatenate((first_row, second_row), axis=0)
        self.output = np.tile(
            comb_first_second_rows, (total_boxes_for_one_row_in_pair, 1)
        )
        return np.copy(self.output)


class Spectrum(Pattern):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.output = None

    def draw(self) -> np.ndarray:
        '''
        The First channel of the 3D array is R, 2nd Channel is Green and 3rd is
        Blue. 
        Our desired spectrum requires densed red color on the right, densed Green
        Color on bottom and Densed Blue color on the left.
        '''
        spectrum_array = np.zeros([self.resolution, self.resolution, 3])
        ##Creates a red channed which have densed red color at the right
        spectrum_array[:, :, 0] = np.linspace(0, 1, self.resolution)
        ##Creates a green channed which have densed green color at the bottom
        spectrum_array[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(
            self.resolution, 1
        )
        ##Creates a blue channed which have densed blue color at the left
        spectrum_array[:, :, 2] = np.linspace(1, 0, self.resolution)
        self.output = spectrum_array

        return np.copy(self.output)
