import numpy as np
from utils import peak_local_max, fft_convolve


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

    def fun(v):
        return (v[0] - v[2]) / (2 * v[2] - 4 * v[1] + 2 * v[0])

    dx = fun(img_log[:, 1])
    dy = fun(img_log[1, :])

    return dx, dy


class PMC(object):
    """
    PMC - Particle Mask Correlation, base on
    https://link.springer.com/article/10.1007/BF03181412

    Parameters
    ----------
    ksize : size of particle kernel.

    gauss_sigma :   float,
                    X and Y sigma for Gaussian function.

    min_distance :  int,
                    Minimum number of pixels separating peaks in a region of 2 * min_distance + 1
                    (i.e. peaks are separated by at least min_distance). To find the maximum number of peaks,
                    use min_distance=1.

    threshold_rel : float,
                    Minimum intensity of peaks, calculated as max(image) * threshold_rel.
    """

    def __init__(self, ksize=(5, 5), gauss_sigma=(1.5, 1.5), min_distance=3, threshold_rel=0.6):
        self.kernel = self._create_gauss_mask(ksize=ksize, sigma=gauss_sigma).astype(np.float32)
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def get_positions(self, img, subpixel=False):
        img = img.astype(np.float32) / img.max()

        field = fft_convolve(img, self.kernel)
        coordinates = peak_local_max(field, min_distance=self.min_distance,
                                     threshold_rel=self.threshold_rel).astype('float32')

        # mask shift correction
        coordinates[:, 0] -= 0.5 * (self.kernel.shape[0] - 1)
        coordinates[:, 1] -= 0.5 * (self.kernel.shape[1] - 1)

        if subpixel:
            w, h = img.shape
            shift = 1
            for i, p in enumerate(coordinates):
                if p[0] != 0 and p[0] != w - 1 and p[1] != 0 and p[1] != h - 1:
                    sub_img = field[p[0] - shift:p[0] + shift + 1, p[1] - shift:p[1] + shift + 1]
                    dx, dy = three_point_gaussian_interpolation(sub_img)

                    coordinates[i] += (dx, dy)

        return coordinates

    def _create_gauss_mask(self, ksize=(5, 5), a=(0, 0), sigma=(1.5, 1.5)):
        """
        Create mask as Gaussian function.

        :param ksize: size of mask.
        :param a: position of center.
        :param sigma: Gauss sigma.
        :return: ksize array with values (0, 1]
        """

        kernel = np.zeros(ksize)

        a = np.array(a)
        s = np.array(sigma)

        def f(arg):
            return np.exp(-np.sum(((a - arg) ** 2) / (2 * s ** 2)))

        for x in range(ksize[0]):
            for y in range(ksize[1]):
                kernel[x, y] = f(np.array([x - np.floor(ksize[0] / 2), y - np.floor(ksize[1] / 2)]))

        kernel /= kernel.max()

        return kernel
