"""Baseline."""
import math
import typing as ty

import numpy as np
from scipy.ndimage.filters import gaussian_filter


# TODO: should try to speed this up as the for-loop makes this very computationally expensive
def curve(data, window: int, **_kwargs):
    """Based on massign method: https://pubs.acs.org/doi/abs/10.1021/ac300056a.

    We initialise an array which has the same length as the input array, subsequently smooth the spectrum  and then
    subtract the background from the spectrum

    Parameters
    ----------
    data: np.array
        intensity array
    window: int
        window integer

    Returns
    -------
    data: np.array
        data without background

    """
    window = abs(window)
    if window <= 0:
        raise ValueError("Value should be above 0")

    return _curve(data, window)


def _curve(data, window: int):
    length = data.shape[0]
    mins = np.zeros(length, dtype=np.int32)
    for i in range(length):
        mins[i] = np.amin(data[int(max([0, i - window])) : int(min([i + window, length]))])
    background = gaussian_filter(mins, window * 2)
    return data - background


def polynomial(y: np.ndarray, deg: int = 4, max_iter: int = 100, tol: float = 1e-3, **_kwargs):
    """
    Taken from: https://peakutils.readthedocs.io/en/latest/index.html
    -----------------------------------------------------------------
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
    max_iter : int (default: 100)
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
    import scipy.linalg

    # for not repeating ourselves in `envelope`
    if deg is None:
        deg = 4
    if max_iter is None:
        max_iter = 100
    if tol is None:
        tol = 1e-3

    order = deg + 1
    coeffs = np.ones(order)

    # try to avoid numerical issues
    cond = math.pow(abs(y).max(), 1.0 / order)
    x = np.linspace(0.0, cond, y.size)
    base = y.copy()

    vander = np.vander(x, order)
    vander_pinv = scipy.linalg.pinv2(vander)

    for _ in range(max_iter):
        coeffs_new = np.dot(vander_pinv, y)

        if scipy.linalg.norm(coeffs_new - coeffs) / scipy.linalg.norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        base = np.dot(vander, coeffs)
        y = np.minimum(y, base)

    return base


def als(y, lam, p, niter=10):
    """Asymmetric Least Squares smoothing. There are two parameters p for asymmetry and lambda for smoothness.
    Values of p should range between 0.001 and 0.1 and lambda between 10^2 to 10^9.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    # taken from: https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    n = len(y)
    diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    w = np.ones(n)
    z = np.zeros_like(w)
    while niter > 0:
        W = sparse.spdiags(w, 0, n, n)
        Z = W + lam * diag.dot(diag.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        niter -= 1
    return z


def arpls(y: np.ndarray, lam: float = 100, ratio: float = 1e-6, niter: int = 10):
    """Modified ALS.

    See: https://stackoverflow.com/a/67509948
    """
    from numpy.linalg import norm
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            break
    return z


def linear(data, threshold: float, **_kwargs):
    """Subtract baseline using linear method."""
    if not isinstance(threshold, (int, float)):
        raise TypeError("Threshold must be a number")
    if threshold < 0:
        raise ValueError("Value should be above 0")

    data[data <= threshold] = 0

    return data


def median(data, median_window: int = 5, **_kwargs):
    """Median-filter."""
    from scipy.ndimage import median_filter

    if median_window % 2 == 0:
        raise ValueError("Median window must be an odd number")

    data = median_filter(data, median_window)
    return data


def tophat(data, tophat_window=100, **_kwargs):
    """Top-hat filter."""
    from scipy.ndimage.morphology import white_tophat

    return white_tophat(data, tophat_window)


def baseline_1d(
    y,
    baseline_method: str = "linear",
    threshold: ty.Optional[float] = None,
    poly_order: ty.Optional[int] = 4,
    max_iter: ty.Optional[int] = 100,
    tol: ty.Optional[float] = 1e-3,
    median_window: ty.Optional[int] = 5,
    curved_window: ty.Optional[int] = None,
    tophat_window: ty.Optional[int] = 100,
    **_kwargs,
):
    """Subtract baseline from the y-axis intensity array.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    baseline_method : str
        baseline removal method
    threshold : float
        any value below `threshold` will be set to 0
    poly_order : int
        Degree of the polynomial that will estimate the data baseline. A low degree may fail to detect all the
        baseline present, while a high degree may make the data too oscillatory, especially at the edges; only used
        with method being `Polynomial`
    max_iter : int
        Maximum number of iterations to perform; only used with method being `Polynomial`
    tol : float
        Tolerance to use when comparing the difference between the current fit coefficients and the ones from the
        last iteration. The iteration procedure will stop when the difference between them is lower than *tol*.; only
        used with method being `Polynomial`
    median_window : int
        median filter size - should be an odd number; only used with method being `Median`
    curved_window : int
        curved window size; only used with method being `Curved`
    tophat_window : int
        tophat window size; only used with method being `Top Hat`

    Returns
    -------
    y : np.ndarray
        y-axis intensity array with baseline removed
    """
    # ensure data is in 64-bit format
    baseline_method = baseline_method.lower()
    y = np.array(y, dtype=np.float64)
    if baseline_method == "linear":
        y = linear(y, threshold=threshold)
    elif baseline_method == "polynomial":
        baseline = polynomial(y, deg=poly_order, max_iter=max_iter, tol=tol)
        y = y - baseline
    elif baseline_method == "median":
        y = median(y, median_window=median_window)
    elif baseline_method == "curved":
        y = curve(y, curved_window)
    elif baseline_method == "tophat":
        y = tophat(y, tophat_window)

    y[y <= 0] = 0

    return y
