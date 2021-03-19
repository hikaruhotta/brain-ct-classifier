"""
dataset_utils.py
Utility functions for the dataset files. This contains functions that have little to do with the dataset class
but are necessary for our exact problem. For example, computations about image boundaries and follow-up cropping
of said images.
"""

import numpy as np


def get_final_bounding_box(bounding_box_1, bounding_box_2):
    """
    Gets bounding box that is contained in both bounding box 1 and 2

    Args:
        bounding_box_1 (np.ndarray of shape (3,2)): [[D_min, D_max], [H_min,, H_max], [W_min, W_max]]
        bounding_box_2 (np.ndarray of shape (3,2)): [[D_min, D_max], [H_min,, H_max], [W_min, W_max]]
    """

    bounding_box_out = np.empty((3, 2))
    for i in range(3):
        bounding_box_out[i][0] = max(bounding_box_1[i][0], bounding_box_2[i][0])  # max of lower bounds
        bounding_box_out[i][1] = min(bounding_box_1[i][1], bounding_box_2[i][1])  # min of upper bounds
    return bounding_box_out


def crop_image_to_new_bounding_box(img, bounding_box_original, bounding_box_new):
    D, H, W = img.shape

    dmin, dmax = bounding_box_original[0]
    hmin, hmax = bounding_box_original[1]
    wmin, wmax = bounding_box_original[2]

    dlen, hlen, wlen = dmax - dmin, hmax - hmin, wmax - wmin

    shifted_box = bounding_box_new - np.array([dmin, hmin, wmin]).reshape((3, 1))
    normed_box = np.divide(shifted_box, np.array([dlen, hlen, wlen]).reshape((3, 1)))
    pixel_cutoffs = np.multiply(normed_box, np.array([D, H, W]).reshape((3, 1))).astype(int)

    d1, d2 = pixel_cutoffs[0]
    h1, h2 = pixel_cutoffs[1]
    w1, w2 = pixel_cutoffs[2]

    return img[d1:d2, h1:h2, w1:w2]
