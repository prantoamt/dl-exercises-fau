# Pyton imports
from typing import Optional, Union

# Third party imports
from torch import nn

# Self imports


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        max_pool: Optional[Union[tuple, int]] = None,
        skip_conn: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.skip_conn = skip_conn
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(max_pool) if max_pool else None

    def forward(self, x) -> None:
        x = self.conv(x)
        x = self.batch(x) + x if self.skip_conn else self.batch(x)
        x = self.max_pool((self.relu(x))) if self.max_pool else self.relu(x)
        return x
