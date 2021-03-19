"""
transforms.py
Modified versions of PyTorch's torchvision transform for lists (3D).
Take in a list of PIL Images and apply the same transform to each.
Assume all the images in one sequence are the same size.
"""

import torch
import torchvision.transforms.functional as F
import numpy as np
from scipy.ndimage import zoom


class ResampleTo:

    def __init__(self, output_shape):
        self.output_shape = tuple(output_shape)

    def __call__(self, np_arr):
        if self.output_shape == np_arr.shape:
            return np_arr

        zoom_amounts = [out_dim / cur_dim for out_dim, cur_dim in zip(self.output_shape, np_arr.shape)]
        reshaped = zoom(np_arr, zoom_amounts)
        assert reshaped.shape == self.output_shape
        return reshaped

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomConsecutiveSlices:

    def __init__(self, num_slices):
        self.num_slices = num_slices

    def __call__(self, np_arr):
        total_slices = np_arr.shape[0]

        if total_slices < self.num_slices:
            raise RuntimeError(
                'RandomConsecutiveSlices given array with too few slices (%s slices wanted, %s in array).' % (
                    self.num_slices, total_slices))
        elif total_slices == self.num_slices:
            return np_arr
        else:
            offset = np.random.randint(0, total_slices - self.num_slices)
            return np_arr[offset:offset + self.num_slices]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class EltListToBlockTensor:
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (D x H x W) in an arbitrary range
    to a torch.FloatTensor of shape (1 x D x H x W).
    """

    def __call__(self, elt_list):
        """
        Args:
            elt_list (numpy.ndarray): 3d volume to be converted to tensor.
        Returns:
            Tensor: Converted 3d volume.
        """

        elt_list = elt_list.astype('float32')
        tensor = torch.from_numpy(elt_list).unsqueeze(0)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToFloat():
    """Converts an ndarray volume from its usual input (int32, uint16, etc) to a float32

    """

    def __call__(self, input_volume):
        """
        Args:
            input_volume (ndarray): ndarray volume of size (D, H, W) to be normalized.
        Returns:
            Tensor: input volume as floats
        """

        return input_volume.astype('float32')


class Normalize():
    """Normalize a ndarray volume
    Uses a hard cutoff at min and max bound to choose where

    """

    # pixel mean was determined by a training dataset runthrough and averaging, after already clipping bounds
    def __init__(self, input_bounds, pixel_mean):
        self.input_bounds = input_bounds
        self.pixel_mean = pixel_mean

    def __call__(self, input_volume):
        """
        Args:
            input_volume (ndarray): ndarray volume of size (D, H, W) to be normalized.
        Returns:
            Tensor: normalized ndarray volume
        """
        input_min, input_max = self.input_bounds

        x = input_volume
        x -= input_min
        x /= (input_max - input_min)
        x = np.maximum(x, 0)  # Winsorizing against low and high density signals that we don't care about
        x = np.minimum(x, 1)

        x -= self.pixel_mean  # theoretically [-1: 1], realistically -.374 to .626

        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(input_bounds={self.input_bounds}, pixel_mean={self.pixel_mean})'
