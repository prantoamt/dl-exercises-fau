# Python imports
from typing import Optional, Union

# Third party imports
from torch import nn
import torch

# Self imports
from conv_block import ConvBlock
from res_block import ResBlock
from providers import (
    conv_block,
    res_block_1,
    res_block_2,
    res_block_3,
    res_block_4,
    flatten,
    sigmoid,
    fully_connected,
    dropout,
)


class ResNet(nn.Module):
    """
    Output of Conv2D can be calculated as:
        output_image_size =  [{(input_image_height_or_weight - karnel_size) + 2 * padding} / stride] +1
    """

    def __init__(
        self,
        conv_block: ConvBlock = conv_block,
        res_block_1: ResBlock = res_block_1,
        res_block_2: ResBlock = res_block_2,
        res_block_3: ResBlock = res_block_3,
        res_block_4: ResBlock = res_block_4,
        flatten: nn.Flatten = flatten,
        fully_connected: nn.Linear = fully_connected,
        sigmoid: nn.Sigmoid = sigmoid,
        dropout: nn.Dropout = dropout,
    ):
        super(ResNet, self).__init__()
        # 1 image = 3 channel * 300 height * 300 width
        self.conv_block = conv_block
        self.res_block_1 = res_block_1
        self.res_block_2 = res_block_2
        self.res_block_3 = res_block_3
        self.res_block_4 = res_block_4
        self.flatten = flatten
        self.fully_connected = fully_connected
        self.sigmoid = sigmoid
        self.dropout = dropout

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_block(x)
        # x = self.dropout(x)
        x = self.res_block_1(x)
        # x = self.dropout(x)
        x = self.res_block_2(x)
        # x = self.dropout(x)
        x = self.res_block_3(x)
        # x = self.dropout(x)
        x = self.res_block_4(x)
        # x = self.dropout(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.flatten(x)
        x = self.fully_connected(x)
        x = self.sigmoid(x)
        return x
