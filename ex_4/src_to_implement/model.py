# Python imports
from typing import Optional, Union

# Third party imports
from torch import nn
import torch

# Self imports
from conv_block import ConvBlock
from res_block import ResBlock


class ResNet(nn.Module):
    """
    Output of Conv2D can be calculated as:
        output_image_size =  [{(input_image_height_or_weight - karnel_size) + 2 * padding} / stride] +1
    """

    def __init__(
        self,
        conv_block: ConvBlock = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=0,
            max_pool=(3, 2),
        ),
        res_block_1: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                max_pool=None,
            ),
            conv_block_2=ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                max_pool=None,
            ),
        ),
        res_block_2: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                max_pool=None,
            ),
            conv_block_2=ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                max_pool=None,
            ),
            downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
            ),
        ),
        res_block_3: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                max_pool=None,
            ),
            conv_block_2=ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                max_pool=None,
            ),
            downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
            ),
        ),
        res_block_4: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                max_pool=None,
            ),
            conv_block_2=ConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                max_pool=None,
            ),
            downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
            ),
        ),
        flatten: nn.Flatten = nn.Flatten(start_dim=1),
        fully_connected: nn.Linear = nn.Linear(
            in_features=512 * 10 * 10, out_features=2
        ),
        sigmoid: nn.Sigmoid = nn.Sigmoid(),
    ):
        super(ResNet, self).__init__()
        # 1 image = 3 channel * 300 height * 300 width
        self.conv_block = conv_block
        self.res_block_1 = res_block_1
        self.res_block_2 = res_block_2
        self.res_block_3 = res_block_3
        self.res_block_4 = res_block_4
        ## TODO: add GlobalAvgPool
        self.flatten = flatten
        self.fully_connected = fully_connected
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        x = self.sigmoid(x)
        return x
