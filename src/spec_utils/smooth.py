"""Smooth."""
import numpy as np
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
    """
    if issparse(ys):
        ys = np.array(ys.todense())

    for i, y in enumerate(ys):
        ys[i] = gaussian(y, sigma)
    return ys
