# Python imports
from typing import Optional, Union

# Third party imports
from torch import nn
import torch.functional as F

# Self imports
from conv_block import ConvBlock


class ResNet(nn.Module):
    def __init__(
        self,
        conv_block: Optional[ConvBlock] = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=1,
            max_pool=(3, 2),
        ),
    ):
        super(ResNet, self).__init__()
        # 1 image = 3 channel * 300 height * 300 width
        self.conv_block_1 = conv_block.get_conv_block()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_block_1(x)
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
