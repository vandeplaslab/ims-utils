"""Various functions to convert centroid peaks to a profile spectrum."""

from __future__ import annotations

import typing as ty

import numpy as np
from koyo.utilities import rescale
from numba import njit

from ims_utils.spectrum import ppm_to_delta_mass

try:
    from ms_peak_picker import FittedPeak
    from ms_peak_picker.peak_statistics import GaussianModel
    from ms_peak_picker.reprofile import PeakSetReprofiler

    HAS_MPP = True
except ImportError:
    HAS_MPP = False

_TWO_SQRT2LN2 = 2.3548200450309493  # FWHM = this * sigma


def _ensure_1d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return a


def _compute_sigma(mz, *, resolving_power=None, fwhm=None, fwhm_func=None):
    """Return a sigma array (same length as mz).

    You can specify ONE of:
      - resolving_power (scalar or array): FWHM_i = mz_i / R_i
      - fwhm (scalar or array): constant or per-peak FWHM
      - fwhm_func (callable): FWHM_i = f(mz_i) (computed in Python).
    """
    mz = _ensure_1d(mz, "mz")

    specified = sum(x is not None for x in (resolving_power, fwhm, fwhm_func))
    if specified != 1:
        raise ValueError("Specify exactly one of resolving_power, fwhm, or fwhm_func.")

    if resolving_power is not None:
        R = np.asarray(resolving_power)
        if R.ndim == 0:
            fwhm_arr = mz / float(R)
        else:
            R = _ensure_1d(R, "resolving_power")
            if R.size != mz.size:
                raise ValueError("resolving_power array must match mz length")
            fwhm_arr = mz / R
    elif fwhm is not None:
        f = np.asarray(fwhm)
        if f.ndim == 0:
            fwhm_arr = np.full_like(mz, float(f))
        else:
            fwhm_arr = _ensure_1d(f, "fwhm")
            if fwhm_arr.size != mz.size:
                raise ValueError("fwhm array must match mz length")
    else:
        # fwhm_func
        fwhm_arr = np.asarray([float(fwhm_func(x)) for x in mz], dtype=float)

    sigma = fwhm_arr / _TWO_SQRT2LN2
    return sigma


def _auto_grid(mz, sigma, window, points_per_fwhm=8, mz_min=None, mz_max=None, bin_width=None):
    if mz_min is None:
        mz_min = float(np.min(mz - window * sigma))
    if mz_max is None:
        mz_max = float(np.max(mz + window * sigma))
    if mz_max <= mz_min:
        raise ValueError("mz_max must be > mz_min")

    if bin_width is None:
        # Use median FWHM / points_per_fwhm as a sensible linear step
        fwhm_med = float(np.median(sigma) * _TWO_SQRT2LN2)
        step = fwhm_med / float(points_per_fwhm)
    else:
        step = float(bin_width)

    n = int(np.floor((mz_max - mz_min) / step)) + 1
    grid = mz_min + step * np.arange(n, dtype=float)
    return grid


@njit(cache=True, fastmath=True)
def _add_peaks_numba(
    mz_grid: np.ndarray, y: np.ndarray, mz: np.ndarray, amp: np.ndarray, sigma: np.ndarray, window: float
):
    n = mz.size
    for i in range(n):
        s = sigma[i]
        mu = mz[i]
        a = amp[i]
        if s <= 0.0 or a == 0.0:
            continue
        left = mu - window * s
        right = mu + window * s

        # searchsorted equivalents
        # numba supports np.searchsorted on 1D arrays
        start = np.searchsorted(mz_grid, left)
        end = np.searchsorted(mz_grid, right)

        if start < 0:
            start = 0
        if end > mz_grid.size:
            end = mz_grid.size

        for j in range(start, end):
            dx = mz_grid[j] - mu
            y[j] += a * np.exp(-0.5 * (dx * dx) / (s * s))


def centroid_to_profile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sort: bool = False,
    # Resolution controls (pick one): resolving_power OR fwhm OR fwhm_func
    resolving_power: float | np.ndarray | None = None,
    fwhm: float | np.ndarray | None = None,
    fwhm_func: ty.Callable | None = None,
    # Grid controls
    mz_grid: np.ndarray | None = None,
    mz_min: float | None = None,
    mz_max: float | None = None,
    bin_width: float | None = None,
    points_per_fwhm: int = 8,
    # Peak rendering
    window: float = 5.0,
    intensity_mode: ty.Literal["height", "area"] = "height",  # "height" or "area"
    dtype: np.dtype = np.float64,
):
    """
    Convert centroid peaks (mz, intensity) to a summed Gaussian profile.

    Parameters
    ----------
    x : (N,) array
        Centroid m/z positions.1
    y : (N,) array
        Centroid intensities. Interpreted per `intensity_mode`.
    sort : bool, default False
        Sort input peaks by m/z if True (otherwise must be pre-sorted).
    resolving_power : float or (N,) array, optional
        If set: FWHM_i = mz_i / R_i. (Specify exactly one of these three:)
    fwhm : float or (N,) array, optional
        Constant or per-peak FWHM (in Da).
    fwhm_func : callable, optional
        Function f(mz) -> FWHM in Da (computed in Python).
    mz_grid : (M,) array, optional
        If provided, render onto this grid (must be sorted ascending).
    mz_min, mz_max : float, optional
        Grid bounds (used only if mz_grid is None).
    bin_width : float, optional
        Linear grid spacing in Da (used only if mz_grid is None).
    points_per_fwhm : int, default 8
        If auto-grid, target ~this many points across median FWHM.
    window : float, default 5.0
        Truncate each Gaussian at ±window * sigma.
    intensity_mode : {"height","area"}, default "height"
        "height": centroid intensities are peak heights (amplitudes).
        "area": centroid intensities are areas; code converts to Gaussian height.
    dtype : numpy dtype, default float64
        Output array dtype.

    Returns
    -------
    mz_profile : (M,) array
        m/z grid.
    y_profile : (M,) array
        Summed profile intensities on the grid.
    sigma_per_peak : (N,) array
        The sigma used for each centroid peak (for reference/QA).

    Notes
    -----
    - Overlapping/nearby peaks are handled by linear superposition.
    - For mass resolution control:
        * Constant resolving power R -> FWHM_i = mz_i / R.
        * Constant FWHM -> same width everywhere.
        * fwhm_func(mz) -> fully custom instrument model.
    """
    x = _ensure_1d(x, "mz").astype(float)
    y = _ensure_1d(y, "intensity").astype(float)
    if x.size != y.size:
        raise ValueError("mz and intensity must have the same length")
    if x.size == 0:
        if mz_grid is None:
            raise ValueError("Empty input and no grid specified.")
        return np.asarray(mz_grid, dtype=float), np.zeros_like(mz_grid, dtype=dtype), np.array([], dtype=float)

    # Sort centroids by m/z (helps windowing/searchsorted)
    if sort:
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    # Per-peak sigma from resolution model
    sigma = _compute_sigma(
        x,
        resolving_power=resolving_power,
        fwhm=fwhm,
        fwhm_func=fwhm_func,
    )

    # Grid
    if mz_grid is None:
        mz_grid = _auto_grid(
            x, sigma, window, points_per_fwhm=points_per_fwhm, mz_min=mz_min, mz_max=mz_max, bin_width=bin_width
        )
    else:
        mz_grid = np.asarray(mz_grid, dtype=float)
        if not np.all(np.diff(mz_grid) > 0):
            raise ValueError("mz_grid must be strictly increasing")

    # Convert intensity to Gaussian amplitude if needed
    if intensity_mode == "height":
        amp = y.astype(float)
    elif intensity_mode == "area":
        amp = y / (sigma * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError("intensity_mode must be 'height' or 'area'")

    # Render
    yy = np.zeros(mz_grid.size, dtype=dtype)
    _add_peaks_numba(mz_grid, yy, x, amp, sigma, float(window))
    return mz_grid, yy, sigma


def centroid_to_profile_ppm(
    x: np.ndarray, y: np.ndarray, mz_min: float, mz_max: float, ppm: float = 1.0, mz: float = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Convert centroid data to profile data.

    Parameters
    ----------
    x : np.ndarray
        Centroid m/z values.
    y : np.ndarray
        Intensities.
    mz_min : float
        Minimum m/z value for the profile spectrum.
    mz_max : float
        Maximum m/z value for the profile spectrum.
    ppm : float
        Spacing in ppm for the profile spectrum.
    mz : float
        M/z value at which the ppm spacing is calculated.
    """
    x = _ensure_1d(x, "mz").astype(float)
    y = _ensure_1d(y, "intensity").astype(float)
    if x.size != y.size:
        raise ValueError("mz and intensity must have the same length")

    # calculate appropriate peak widths for each centroid location
    widths = ppm_to_delta_mass(x, ppm)
    # convert peaks to profile mode
    models = [
        GaussianModel(FittedPeak(x[index], y[index], 0, 0, 0, widths[index], y[index])) for index in range(x.size)
    ]
    models.insert(0, GaussianModel(FittedPeak(mz_min, 0, 0, 0, 0, 0.01, 0)))
    models.append(GaussianModel(FittedPeak(mz_max, 0, 0, 0, 0, 0.01, 0)))
    task = PeakSetReprofiler(models, dx=ppm_to_delta_mass(mz, ppm))
    xx, yy = task.reprofile()
    yy /= len(models)
    # # re-profile the data
    # xx, yy = reprofile(models, dx=ppm_to_delta_mass(self.ppm_at_mz, self.ppm))
    # rescale the data so that it matches perfectly with the centroid data
    yy = rescale(yy, 0, y.max())
    return xx, yy
