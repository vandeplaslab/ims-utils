"""Peakutils code.

This code is copied from the peakutils source code.

Code: https://bitbucket.org/lucashnegri/peakutils/src/master/

"""
import math
import typing as ty

import numpy as np
import scipy.linalg as LA
from scipy import optimize
from scipy.integrate import simps

eps = np.finfo(float).eps


def baseline(y, deg=3, max_it=100, tol=1e-3):
    """
    Computes the baseline of a given data.

    Iteratively performs a polynomial fitting in the data to detect its
    baseline. At every iteration, the fitting weights on the regions with
    peaks are reduced to identify the baseline only.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int (default: 3)
        Degree of the polynomial that will estimate the data baseline. A low
        degree may fail to detect all the baseline present, while a high
        degree may make the data too oscillatory, especially at the edges.
    max_it : int (default: 100)
        Maximum number of iterations to perform.
    tol : float (default: 1e-3)
        Tolerance to use when comparing the difference between the current
        fit coefficients and the ones from the last iteration. The iteration
        procedure will stop when the difference between them is lower than
        *tol*.

    Returns
    -------
    ndarray
        Array with the baseline amplitude for every original point in *y*
    """
    # for not repeating ourselves in `envelope`
    order = deg + 1
    coeffs = np.ones(order)

    # try to avoid numerical issues
    cond = math.pow(abs(y).max(), 1.0 / order)
    x = np.linspace(0.0, cond, y.size)
    base = y.copy()

    vander = np.vander(x, order)
    vander_pinv = LA.pinv(vander)

    for _ in range(max_it):
        coeffs_new = np.dot(vander_pinv, y)

        if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        base = np.dot(vander, coeffs)
        y = np.minimum(y, base)

    return base


def envelope(y, deg=None, max_it=None, tol=None):
    """
    Computes the upper envelope of a given data.
    It is implemented in terms of the `baseline` function.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int
        Degree of the polynomial that will estimate the envelope.
    max_it : int
        Maximum number of iterations to perform.
    tol : float
        Tolerance to use when comparing the difference between the current
        fit coefficients and the ones from the last iteration.

    Returns
    -------
    ndarray
        Array with the envelope amplitude for every original point in *y*
    """
    return y.max() - baseline(y.max() - y, deg, max_it, tol)


def indexes(y: np.ndarray, thres: float = 0.3, min_dist: int = 1, thres_abs: bool = False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    (zeros,) = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        (zeros_diff_not_one,) = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & (np.greater(y, thres)))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]
    return peaks


def centroid2(y: np.ndarray, x: ty.Optional[np.ndarray] = None, dx: float = 1.0):
    """Computes the centroid for the specified data.
    Not intended to be used.

    Parameters
    ----------
    y : array_like
        Array whose centroid is to be calculated.
    x : array_like, optional
        The points at which y is sampled.

    Returns
    -------
    (centroid, sd)
        Centroid and standard deviation of the data.
    """
    yt = np.array(y)

    if x is None:
        x = np.arange(yt.size, dtype="float") * dx

    normaliser = simps(yt, x)
    centroid = simps(x * yt, x) / normaliser
    var = simps((x - centroid) ** 2 * yt, x) / normaliser
    return centroid, np.sqrt(var)


def gaussian(x: float, ampl: float, center: float, dev: float):
    """Computes the Gaussian function.

    Parameters
    ----------
    x : number
        Point to evaluate the Gaussian for.
    a : number
        Amplitude.
    b : number
        Center.
    c : number
        Width.

    Returns
    -------
    float
        Value of the specified Gaussian at *x*
    """
    return ampl * np.exp(-((x - float(center)) ** 2) / (2.0 * dev ** 2 + eps))


def gaussian_fit(x: np.ndarray, y: np.ndarray, center_only: bool = True):
    """Performs a Gaussian fitting of the specified data.

    Parameters
    ----------
    x : ndarray
        Data on the x axis.
    y : ndarray
        Data on the y axis.
    center_only: bool
        If True, returns only the center of the Gaussian for `interpolate` compatibility

    Returns
    -------
    ndarray or float
        If center_only is `False`, returns the parameters of the Gaussian that fits the specified data
        If center_only is `True`, returns the center position of the Gaussian
    """
    if len(x) < 3:
        # used RuntimeError to match errors raised in scipy.optimize
        raise RuntimeError("At least 3 points required for Gaussian fitting")

    initial = [np.max(y), x[0], (x[1] - x[0]) * 5]
    params, pcov = optimize.curve_fit(gaussian, x, y, initial)

    if center_only:
        return params[1]
    else:
        return params
