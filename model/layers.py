"""
Layers for generator and discriminator
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    # TODO use second to last and last instead of 3rd and 4th (this makes code general between 2D and 3D)
    middle_height = image.shape[3] // 2
    middle_width = image.shape[4] // 2
    starting_height = middle_height - new_shape[3] // 2
    final_height = starting_height + new_shape[3]
    starting_width = middle_width - new_shape[4] // 2
    final_width = starting_width + new_shape[4]
    cropped_image = image[:, :, :, starting_height:final_height, starting_width:final_width]
    return cropped_image


class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm3d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear',
                                    align_corners=True)  # TODO Use bicubic instead of trilinear?
        self.conv1 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels // 2, input_channels // 2, kernel_size=3, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm3d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x


class ReplicationPad3d(nn.Module):
    '''
    ReflectionPad3d Class
    Values
    padding: an int, tuple of 3 ints (symettric), or a tuple of 3 tuples for padding dimensions
    '''

    __constants__ = ['padding']

    def __init__(self, padding):
        super().__init__()
        assert type(padding) is int or type(padding) is tuple
        if type(padding) is tuple:
            assert len(padding) == 3
            self.padding = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        else:
            self.padding = (padding, padding, padding, padding, padding, padding)

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'replicate')

    def extra_repr(self) -> str:
        return '{}'.format(self.padding)


class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
    channels: the number of channels throughout the residual block, a scalar
    '''

    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm3d(channels, affine=False),

            nn.ReLU(inplace=True),

            ReplicationPad3d(1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm3d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.layers(x)


class Pix2PixHDPatchDiscriminator(nn.Module):
    '''
    Discriminator Class
    Implements the discriminator class for a subdiscriminator,
    which can be used for all the different scales, just with different argument values.
    Values:
    in_channels: the number of channels in input, a scalar
    base_channels: the number of channels in first convolutional layer, a scalar
    n_layers: the number of convolutional layers, a scalar
        (discriminators trained with least squares loss should set this to be False)
    '''

    def __init__(self, in_channels=1, base_channels=16, n_layers=3):
        super().__init__()

        # Use nn.ModuleList so we can output intermediate values for loss.
        self.layers = nn.ModuleList()

        # Initial convolutional layer
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Downsampling convolutional layers
        channels = base_channels
        for _ in range(1, n_layers):
            prev_channels = channels
            channels = min(2 * channels, 512)
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(prev_channels, channels, kernel_size=3, stride=(1, 2, 2), padding=1),
                    nn.InstanceNorm3d(channels, affine=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output convolutional layer
        prev_channels = channels
        channels = min(2 * channels, 512)
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(prev_channels, channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(channels, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(channels, 1, kernel_size=3, stride=1, padding=1),
            )
        )

    def forward(self, x):
        outputs = []  # for feature matching loss
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs
