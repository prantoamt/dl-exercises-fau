# Pyton imports
from typing import Optional, Union

# Third party imports
from torch import nn

# Self imports


class ConvBlock:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        max_pool: Optional[Union[tuple, int]] = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool = max_pool

    def get_conv_block(self) -> nn.Sequential:
        layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
        )
        layers.append(nn.MaxPool2d(self.max_pool)) if self.max_pool else None

        return layers
