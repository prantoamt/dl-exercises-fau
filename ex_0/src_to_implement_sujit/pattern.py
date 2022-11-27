import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        no_of_tile = (
            self.resolution // self.tile_size
        )  # calculate the number of tiles required
        black_block = np.zeros(
            (self.tile_size, self.tile_size), dtype=int
        )  # init black block of tile size
        white_block = np.ones(
            (self.tile_size, self.tile_size), dtype=int
        )  # init white block of tile size

        first_row = np.concatenate(
            (black_block, white_block), axis=1
        )  # create a horizontal 1*2 grid (black * white)
        second_row = np.concatenate(
            (white_block, black_block), axis=1
        )  # create a horizontal 1*2 grid (white * black)
        initial_grid = np.concatenate(
            (first_row, second_row)
        )  # concatenate 2 horizontal grid to make it 2*2 grid

        self.output = np.tile(
            initial_grid, (no_of_tile // 2, no_of_tile // 2)
        )  # copy the 2*2 grid for no of tiles/2 times
        # to make the desire output

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.draw(), cmap="gray")
        plt.suptitle("Checkerboard")
        plt.xticks([])
        plt.yticks([])
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        # init a meshgrid of res*res
        x = np.linspace(0, self.resolution, self.resolution, dtype=int)
        y = np.linspace(0, self.resolution, self.resolution, dtype=int)
        X, Y = np.meshgrid(x, y)

        # find out all the indexes which are inside the circle of desire radius
        center_x, center_y = self.position
        pts_inside_cir = (X - center_x) ** 2 + (Y - center_y) ** 2 <= self.radius**2

        self.output = np.zeros(X.shape)  # init a black grid of res*res
        self.output[
            pts_inside_cir
        ] = 1  # make all the points inside of the desire circle white

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.draw(), cmap="gray")
        plt.suptitle("Binary Circle")
        plt.xticks([])
        plt.yticks([])
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        # init a empty grid of res*res which has 3 channels
        self.output = np.empty((self.resolution, self.resolution, 3))

        # red channel distribution -> right (max) to left (min)
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)

        # green channel distribution -> bottom (max) to top (min)
        self.output[:, :, 1] = np.linspace(
            0, 1, self.resolution
        )  # frist -> make it right to left
        self.output[:, :, 1] = np.transpose(
            self.output[:, :, 1]
        )  # second -> transpose so that it can transform into bottom to top

        # blue channel distribution -> left (max) to right (min)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.draw())
        plt.suptitle("RGB Spectrum")
        plt.xticks([])
        plt.yticks([])
        plt.show()
