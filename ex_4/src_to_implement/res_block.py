# Pyton imports
from typing import Optional

# Third party imports
import torch

# Self imports
from conv_block import ConvBlock


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        conv_block_1: ConvBlock,
        conv_block_2: ConvBlock,
        downsample: Optional[torch.nn.Sequential] = None,
    ) -> None:
        super().__init__()
        self.conv_block_1 = conv_block_1
        self.conv_block_2 = conv_block_2
        self.downsample = downsample

    def forward(self, x) -> torch.tensor:
        res = x if self.downsample == None else self.downsample(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x, res)
        return x
