# Pyton imports
from typing import Optional, Union

# Third party imports
import torch

# Self imports


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        max_pool: Optional[Union[tuple, int]] = None,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool = (
            torch.nn.MaxPool2d(kernel_size=max_pool[0], stride=max_pool[1])
            if max_pool != None
            else None
        )

    def forward(self, x, res: Optional[torch.tensor] = None) -> torch.tensor:
        x = self.conv(x)
        x = self.batch_norm(x) + res if res != None else self.batch_norm(x)
        x = self.max_pool(self.relu(x)) if self.max_pool else self.relu(x)
        return x
