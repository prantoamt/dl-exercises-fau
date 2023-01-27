# Python imports
from typing import Optional, Union

# Third party imports
from torch import nn
import torch.functional as F

# Self imports
from conv_block import ConvBlock
from res_block import ResBlock


class ResNet(nn.Module):
    def __init__(
        self,
        conv_block: ConvBlock = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=0,
            max_pool=(3, 2),
            skip_conn=False,
        ),
        res_block_1: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
            conv_block_2=ConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
        ),
        res_block_2: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
            conv_block_2=ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
        ),
        res_block_3: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
            conv_block_2=ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
        ),
        res_block_4: ResBlock = ResBlock(
            conv_block_1=ConvBlock(
                in_channels=356,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
            conv_block_2=ConvBlock(
                in_channels=356,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                max_pool=None,
                skip_conn=True,
            ),
        ),
        flatten: nn.Flatten = nn.Flatten(start_dim=1),
        fully_connected: nn.Linear = nn.Linear(in_features=512, out_features=2),
    ):
        super(ResNet, self).__init__()
        # 1 image = 3 channel * 300 height * 300 width
        self.conv_block = conv_block
        self.res_block_1 = res_block_1
        self.res_block_2 = res_block_2
        # self.res_block_3 = res_block_3
        # self.res_block_4 = res_block_4
        # an affine operation: y = Wx + b
        self.flatten = flatten
        self.fully_connected = fully_connected

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


tinymodel = ResNet()

print("The model:")
print(tinymodel)
