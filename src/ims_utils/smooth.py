"""Smooth."""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import issparse


def gaussian(y, sigma=1):
    """Smooth 1D spectra using Gaussian filter.

    Parameters
    ----------
    y : np.array
        signal
    sigma : float, optional
        standard deviation of the Gaussian kernel

    Returns
    -------
    y : np.array
        smoothed signal

    Notes
    -----
    The input is not modified; the result has the same shape as ``y``.
    """
    return gaussian_filter(y, sigma=sigma, order=0)


def smooth_spectra(ys, sigma=1):
    """Smooth 2D spectra (one by one) using Gaussian filter.

    Parameters
    ----------
    ys : np.array
        array of signals
    sigma : float, optional
        standard deviation of the Gaussian kernel

    Returns
    -------
    ys : np.array
        smoothed array of signals

    Notes
    -----
    Dense input is smoothed in place and returned. Sparse input is converted to a
    dense array before smoothing and returning.
    """
    if issparse(ys):
        ys = np.array(ys.todense())

    for i, y in enumerate(ys):
        ys[i] = gaussian(y, sigma)
    return ys
