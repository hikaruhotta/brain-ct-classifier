"""
Generators for Generative Adversarial Networks
"""

import torch.nn as nn

from .layers import *


class Pix2PixGenerator(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=1):
        # TODO handle default parameters? Base this off particular dataset?
        super(Pix2PixGenerator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand0 = ExpandingBlock(hidden_channels * 64)
        self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn)


class Pix2PixHDGenerator(nn.Module):
    '''
    Pix2PixHdGenerator Class:
    Implements the local enhancer subgenerator (G2) for handling larger scale images.
    Values:
        in_channels: the number of input channels, a scalar
        out_channels: the number of output channels, a scalar
        base_channels: the number of channels in first convolutional layer, a scalar
        global_fb_blocks: the number of global generator frontend / backend blocks, a scalar
        global_res_blocks: the number of global generator residual blocks, a scalar
        local_res_blocks: the number of local enhancer residual blocks, a scalar
    '''

    def __init__(self, in_channels=1, out_channels=1, base_channels=16, global_fb_blocks=3, global_res_blocks=9,
                 local_res_blocks=3):
        super().__init__()

        global_base_channels = 2 * base_channels

        # Downsampling layer for high-res -> low-res input to g1
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)

        # Initialize global generator without its output layers (the input layers are kept)
        self.g1 = GlobalGenerator(
            in_channels, out_channels, base_channels=global_base_channels, fb_blocks=global_fb_blocks,
            res_blocks=global_res_blocks,
        ).g1

        self.g2 = nn.ModuleList()

        # Initialize local frontend block
        self.g2.append(
            nn.Sequential(
                # Initial convolutional layer
                ReplicationPad3d((1, 3, 3)),
                nn.Conv3d(in_channels, base_channels, kernel_size=(3, 7, 7), padding=0),
                nn.InstanceNorm3d(base_channels, affine=False),
                nn.ReLU(inplace=True),

                # Frontend block
                nn.Conv3d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(2 * base_channels, affine=False),
                nn.ReLU(inplace=True),
            )
        )

        # Initialize local residual and backend blocks
        self.g2.append(
            nn.Sequential(
                # Residual blocks
                *[ResidualBlock(2 * base_channels) for _ in range(local_res_blocks)],

                # Backend blocks
                nn.Upsample(scale_factor=2, mode='trilinear',
                            align_corners=True),  # TODO Use bicubic instead of trilinear?
                nn.Conv3d(2 * base_channels, base_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(base_channels, affine=False),
                nn.ReLU(inplace=True),

                # Output convolutional layer
                ReplicationPad3d((1, 3, 3)),
                nn.Conv3d(base_channels, out_channels, kernel_size=(3, 7, 7), padding=0),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        # Get output from g1_B
        x_g1 = self.downsample(x)
        x_g1 = self.g1(x_g1)

        # Get output from g2_F
        x_g2 = self.g2[0](x)

        # Get final output from g2_B
        return self.g2[1](x_g1 + x_g2)


class GlobalGenerator(nn.Module):
    '''
    GlobalGenerator Class:
    Implements the global subgenerator (G1) for transferring styles at lower resolutions.
    Values:
        in_channels: the number of input channels, a scalar
        out_channels: the number of output channels, a scalar
        base_channels: the number of channels in first convolutional layer, a scalar
        fb_blocks: the number of frontend / backend blocks, a scalar
        res_blocks: the number of residual blocks, a scalar
    '''

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, fb_blocks=3, res_blocks=9):
        super().__init__()

        # Initial convolutional layer
        g1 = [
            ReplicationPad3d((1, 3, 3)),
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 7, 7), padding=0),
            nn.InstanceNorm3d(base_channels, affine=False),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels
        # Frontend blocks
        for _ in range(fb_blocks):
            g1 += [
                nn.Conv3d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(2 * channels, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # Residual blocks
        for _ in range(res_blocks):
            g1 += [ResidualBlock(channels)]

        # Backend blocks
        for _ in range(fb_blocks):
            g1 += [
                nn.Upsample(scale_factor=2, mode='trilinear',
                                            align_corners=True),  # TODO Use bicubic instead of trilinear?
                nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1),
                nn.InstanceNorm3d(channels // 2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        # Output convolutional layer as its own nn.Sequential since it will be omitted in second training phase
        self.out_layers = nn.Sequential(
            ReplicationPad3d((1, 3, 3)),
            nn.Conv3d(base_channels, out_channels, kernel_size=(3, 7, 7), padding=0),
            nn.Tanh(),
        )

        self.g1 = nn.Sequential(*g1)

    def forward(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x

