import numpy as np
import cv2
from skimage.feature import peak_local_max
from .subpixel_peak import three_point_gaussian_interpolation


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
        self.kernel = self._create_gauss_mask(ksize=ksize, sigma=gauss_sigma)
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def get_positions(self, img):
        field = cv2.matchTemplate(img.astype('float32')/img.max(), self.kernel, cv2.TM_CCOEFF)
        coordinates = peak_local_max(field.copy(), min_distance=self.min_distance,
                                     threshold_rel=self.threshold_rel).astype('float32')

        w, h = img.shape
        shift = 1
        for i, p in enumerate(coordinates):
            if p[0] != 0 and p[0] != w - 1 and p[1] != 0 and p[1] != h - 1:
                sub_img = field[p[0]-shift:p[0]+shift+1, p[1]-shift:p[1]+shift+1]
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

        f = lambda x: np.exp(-np.sum(((a - x)**2) / (2 * s**2)))

        for x in range(ksize[0]):
            for y in range(ksize[1]):
                kernel[x, y] = f(np.array([x - np.floor(ksize[0] / 2), y - np.floor(ksize[1] / 2)]))

        kernel /= kernel.max()

        return kernel
