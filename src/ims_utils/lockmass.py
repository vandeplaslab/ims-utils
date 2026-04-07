"""Lockmass functions."""

from __future__ import annotations

import numbers
import typing as ty
import warnings

import numba as nb
import numpy as np
from koyo.utilities import find_nearest_index, get_array_mask

from ims_utils.spectrum import fast_parabolic_centroid

if ty.TYPE_CHECKING:
    try:
        from imzy import BaseReader  # type: ignore[import]
    except ImportError:  # pragma: no cover
        BaseReader = None  # type: ignore[assignment]


# Intensity threshold for lockmass detection
LOCKMASS_THRESHOLD: float = 500


class LockmassEstimator:
    """Lockmass estimator class."""

    def __init__(self, mz_x: np.ndarray, peaks: np.ndarray):
        """Initialize lockmass estimator.

        Parameters
        ----------
        mz_x: np.ndarray
            Mass values of the spectrum. This should be a profile mass spectrum.
        peaks: np.ndarray
            List of lockmass peaks to use for estimation.
        """
        self.mz_x = mz_x
        self.peaks = np.sort(peaks)
        self.n_peaks = len(self.peaks)


    def __call__(self, mz_y: np.ndarray,  **kwargs: ty.Any) -> np.ndarray:
        """Estimate lockmass shifts for the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        """
        return self.estimate(mz_y, **kwargs)

    def estimate(self, mz_y: np.ndarray, **kwargs: ty.Any) -> np.ndarray:
        """Estimate lockmass shifts for the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        """
        raise NotImplementedError("Must implement method")

    @staticmethod
    def apply(mz_y: np.ndarray, shift: int | float) -> np.ndarray:
        """Apply lockmass shift to the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        shift: int | float
            Shift value to apply. Float values are rounded to the nearest integer.
        """
        return fast_roll(mz_y, shift)

    def estimate_for_reader(
        self, reader: BaseReader, weighted: bool = True, silent: bool = False
    ) -> np.ndarray:
        """Estimate lockmass shifts for all spectra in the given reader.

        Parameters
        ----------
        reader : BaseReader
            Reader object to estimate lockmass shifts for. The reader should have a `n_spectra` attribute
            and support indexing to get individual spectra.
        weighted: bool
            Whether to use weighted distance for peak selection.
        silent: bool
            Whether to suppress progress output from the reader iterator.
        """
        if not hasattr(reader, "n_pixels"):
            raise ValueError("reader must have n_pixels attribute")
        if not hasattr(reader, "spectra_iter"):
            raise ValueError("reader must have spectra_iter attribute")
        n_spectra = reader.n_pixels
        shifts = np.zeros((n_spectra, self.n_peaks), dtype=np.float32)
        for i, (_mz_x, mz_y) in enumerate(reader.spectra_iter(silent=silent)):
            self.estimate(mz_y, weighted=weighted, out=shifts[i])
        return shifts


class MaximumIntensityLockmassEstimator(LockmassEstimator):
    """Maximum intensity lockmass estimator."""

    def __init__(self, mz_x: np.ndarray, peaks: np.ndarray, window: float = 0.1):
        """Initialize lockmass estimator.

        Parameters
        ----------
        mz_x: np.ndarray
            Mass values of the spectrum. This should be a profile mass spectrum.
        peaks: np.ndarray
            List of lockmass peaks to use for estimation.
        window: float
            Window size for lockmass peak detection.
        """
        super().__init__(mz_x, peaks)
        self.window = window

        self.mz_indices, self.peak_indices, self.masks, self.offsets, self._starts, self._stops, self._offsets_arr = (
            _prepare_lockmass(mz_x, self.peaks, window)
        )

    def estimate(self, mz_y: np.ndarray, weighted: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """Estimate lockmass shifts for the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        weighted: bool
            Accepted for API compatibility but not used by this estimator; peak is located by
            maximum intensity only.
        out: np.ndarray | None
            Output array to store the results. If None, a new array will be created.
        """
        out = np.zeros(self.n_peaks, dtype=np.float32) if out is None else out
        return _nb_estimate_lockmass_maximum(mz_y, self._starts, self._stops, self._offsets_arr, out)


class WeightedIntensityLockmassEstimator(LockmassEstimator):
    """Weighted intensity lockmass estimator."""

    def __init__(self, mz_x: np.ndarray, peaks: np.ndarray, window: float = 0.1, centroid_frac: float = 0.25):
        """Initialize lockmass estimator.

        Parameters
        ----------
        mz_x: np.ndarray
            Mass values of the spectrum. This should be a profile mass spectrum.
        peaks: np.ndarray
            List of lockmass peaks to use for estimation.
        window: float
            Window size for lockmass peak detection.
        centroid_frac: float
            Fraction of the mean intensity within a peak window used as the centroid peak threshold.
            Peaks below ``mean(y) * centroid_frac`` are ignored during centroid estimation.
        """
        super().__init__(mz_x, peaks)
        self.window = window
        self.centroid_frac = centroid_frac

        self.mz_indices, self.peak_indices, self.masks, _, self._starts, self._stops, self._offsets_arr = (
            _prepare_lockmass(mz_x, self.peaks, window)
        )
        self.centroid_func = fast_parabolic_centroid

    def estimate(self, mz_y: np.ndarray, weighted: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """Estimate lockmass shifts for the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        weighted: bool
            Whether to use weighted distance for peak selection.
        out: np.ndarray | None
            Output array to store the results. If None, a new array will be created.
        """
        out = np.zeros(self.n_peaks, dtype=np.float32) if out is None else out
        peak_max = max(float(np.max(mz_y[s:e])) for s, e in zip(self._starts, self._stops))
        if peak_max < LOCKMASS_THRESHOLD:
            warnings.warn(
                "Spectrum intensities are below the lockmass threshold; returning zero shifts.",
                RuntimeWarning,
                stacklevel=2,
            )
            return out
        return _nb_estimate_lockmass_shifts(
            mz_y,
            self._starts,
            self._stops,
            self.peak_indices,
            self._offsets_arr,
            weighted,
            out,
            LOCKMASS_THRESHOLD,
            LOCKMASS_THRESHOLD * 0.5,
            self.centroid_frac,
        )


def _prepare_lockmass(
    x: np.ndarray, peaks: ty.Iterable[float], window: float = 0.1
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], dict[float, int], np.ndarray, np.ndarray, np.ndarray]:
    """Prepare lockmass data structures.

    Returns
    -------
    mz_indices, peak_indices, masks, offsets, starts, stops, offsets_arr
        The last three are contiguous int64 arrays suitable for numba JIT functions.
        ``starts[j]`` and ``stops[j]`` are the first and one-past-last indices of the j-th
        peak window in ``x``, so ``x[starts[j]:stops[j]]`` is a no-copy view of the window.
        ``offsets_arr[j]`` is the index of the reference peak within that window.
    """
    mz_indices = np.arange(x.shape[0])

    peak_indices = find_nearest_index(x, peaks)
    masks = {peak: get_array_mask(x, peak - window, peak + window) for peak in peaks}
    offsets = {peak: find_nearest_index(x[masks[peak]], peak) for peak in peaks}

    # Precompute contiguous integer slice arrays for numba-compatible access
    starts = np.array([np.where(m)[0][0] for m in masks.values()], dtype=np.int64)
    stops = np.array([np.where(m)[0][-1] + 1 for m in masks.values()], dtype=np.int64)
    offsets_arr = np.array(list(offsets.values()), dtype=np.int64)

    return mz_indices, peak_indices, masks, offsets, starts, stops, offsets_arr


@nb.njit(fastmath=True, cache=True)
def _nb_estimate_lockmass_maximum(
    mz_y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    offsets_arr: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Estimate lockmass shifts using maximum intensity (numba JIT version).

    Uses precomputed slice indices for zero-copy window access.
    """
    for j in range(len(starts)):
        s, e = starts[j], stops[j]
        best_k = 0
        best_val = mz_y[s]
        for k in range(1, e - s):
            v = mz_y[s + k]
            if v > best_val:
                best_val = v
                best_k = k
        out[j] = best_k - offsets_arr[j]
    return out


@nb.njit(fastmath=True, cache=True)
def _nb_parabolic_centroid_window(
    y: np.ndarray,
    global_offset: nb.int64,
    threshold: nb.float64,
) -> tuple[nb.float64, nb.float64]:
    """Find the dominant centroid in a small intensity window using a 3-point parabolic fit.

    This is a numba-native replacement for fast_parabolic_centroid on small windows.
    It finds local maxima above *threshold*, applies a parabolic fit, and returns the
    position and intensity of the centroid closest to the centre of the window.
    Returns (NaN, NaN) if no valid centroid is found.

    Parameters
    ----------
    y : np.ndarray
        Intensity values of the window (slice of the full spectrum).
    global_offset : int
        Index of the first element of this window in the full spectrum. Used so that
        the returned centroid position is in global index coordinates.
    threshold : float
        Minimum peak intensity to consider.
    """
    n = len(y)
    best_cx = np.nan
    best_cy = -1.0
    centre = global_offset + n / 2.0

    for i in range(1, n - 1):
        yi = y[i]
        if yi <= threshold:
            continue
        # Accept strict local maxima and left-edges of plateaus (y[i] == y[i+1] but > y[i-1])
        if yi < y[i - 1] or yi < y[i + 1]:
            continue
        if yi == y[i - 1] and yi == y[i + 1]:
            continue  # true flat region, not a peak
        # 3-point parabolic fit
        x0 = nb.float64(global_offset + i - 1)
        x1 = nb.float64(global_offset + i)
        x2 = nb.float64(global_offset + i + 1)
        y0, y1, y2 = nb.float64(y[i - 1]), nb.float64(yi), nb.float64(y[i + 1])

        dx10 = x1 - x0
        dx21 = x2 - x1
        dx20 = x2 - x0
        if dx10 == 0.0 or dx21 == 0.0 or dx20 == 0.0:
            continue
        a = ((y2 - y1) / dx21 - (y1 - y0) / dx10) / dx20
        if a == 0.0:
            continue
        b = (y2 - y1) / dx21 * dx10 / dx20 + (y1 - y0) / dx10 * dx21 / dx20
        cx = 0.5 * (-b + 2.0 * a * x1) / a
        cy = a * (cx - x1) ** 2 + b * (cx - x1) + y1

        if np.isnan(cx) or np.isnan(cy):
            continue
        # prefer the centroid closest to the window centre
        if best_cx is np.nan or abs(cx - centre) < abs(best_cx - centre):
            best_cx = cx
            best_cy = cy

    return best_cx, best_cy


@nb.njit(fastmath=True, cache=True)
def _nb_estimate_lockmass_shifts(
    mz_y: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    peak_indices: np.ndarray,
    offsets_arr: np.ndarray,
    weighted: nb.boolean,
    out: np.ndarray,
    threshold: nb.float64,
    individual_threshold: nb.float64,
    centroid_frac: nb.float64,
) -> np.ndarray:
    """Estimate lockmass shifts using parabolic centroid fitting (numba JIT version).

    Uses precomputed slice indices for zero-copy window access and an inline
    numba-native centroid finder that avoids scipy overhead.
    """
    n_peaks = len(starts)

    # Check global intensity gate first
    max_intensity = -1.0
    for j in range(n_peaks):
        s, e = starts[j], stops[j]
        for k in range(s, e):
            if mz_y[k] > max_intensity:
                max_intensity = mz_y[k]
    if max_intensity < threshold:
        return out  # caller emits warning at Python level

    for j in range(n_peaks):
        s, e = starts[j], stops[j]
        index = peak_indices[j]

        # Per-peak intensity gate
        peak_max = -1.0
        for k in range(s, e):
            if mz_y[k] > peak_max:
                peak_max = mz_y[k]
        if peak_max < individual_threshold:
            continue

        y = mz_y[s:e]
        win_threshold = np.mean(y) * centroid_frac
        cx, cy = _nb_parabolic_centroid_window(y, s, win_threshold)

        if np.isnan(cx):
            continue
        out[j] = cx - index
    return out



def fast_roll(array: np.ndarray, num: int | float, fill_value: int | float = 0) -> np.ndarray:
    """Shift 1d array to a new position with 0 padding to prevent wraparound - this function is actually
    quicker than np.roll.

    Parameters
    ----------
    array : np.ndarray
        array to be shifted
    num : int | float
        value by which the array should be shifted. Float values are rounded to the nearest integer.
    fill_value : Union[float, int]
        value to fill in the areas where wraparound would have happened
    """
    if not isinstance(num, (numbers.Integral, float, np.floating)):
        raise ValueError("`num` must be a numeric value")
    num = int(round(num))

    if num == 0:
        return array

    result = np.empty_like(array)
    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    return result
