from torch import nn

from conv_block import ConvBlock
from res_block import ResBlock


conv_block = ConvBlock(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=0,
    max_pool=(3, 2),
)

res_block_1 = ResBlock(
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
)

res_block_2 = ResBlock(
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
)

res_block_3 = ResBlock(
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
)

res_block_4 = ResBlock(
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
)

flatten = nn.Flatten(start_dim=1)

fully_connected = nn.Linear(in_features=512 * 10 * 10, out_features=2)

sigmoid = nn.Sigmoid()
