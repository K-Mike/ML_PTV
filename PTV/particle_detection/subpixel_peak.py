"""
The functions of the refining positions of the peak center.
"""

import numpy as np


def three_point_gaussian_interpolation(img):
    """
    Specify the coordinates of the peak based on the maximum sample x of a local image area
    or the correlation function and the two neighbors x + 1
    and x - 1. https://link.springer.com/article/10.1007/s00348-005-0942-3

    :param img: numpy array with size (3, 3).
    Don't send an image with the same values, it will return (nan, nan).
    :return: Refinement of coordinates (dx, dy).
    """

    if img.shape != (3, 3):
        print('three_point_gaussian_interpolation: not correct input image shape, '
              'input image shape must be (3, 3).')
        return

    img_log = np.log(img)

    fun = lambda v: (v[0] - v[2]) / (2 * v[2] - 4 * v[1] + 2 * v[0])

    dx = fun(img_log[:, 1])
    dy = fun(img_log[1, :])

    return dx, dy
