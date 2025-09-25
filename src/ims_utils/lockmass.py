"""Lockmass functions."""

from __future__ import annotations

import typing as ty

import numpy as np
from koyo.utilities import find_nearest_index, get_array_mask

from ims_utils.spectrum import fast_parabolic_centroid

try:
    from imzy import BaseReader  # type: ignore[import]
except ImportError:  # pragma: no cover
    BaseReader = None  # type: ignore[assignment]


# Intensity threshold for lockmass detection
LOCKMASS_THRESHOLD: float = 500


class LockmassEstimator:
    """Lockmass estimator class."""

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
        self.mz_x = mz_x
        self.peaks = peaks
        self.n_peaks = len(peaks)
        self.window = window

        self.mz_indices, self.peak_indices, self.masks = _prepare_lockmass(mz_x, peaks, window)
        self.centroid_func = fast_parabolic_centroid

    def __call__(self, mz_y: np.ndarray, weighted: bool = True, out: np.ndarray | None = None) -> np.ndarray:
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
        return self.estimate(mz_y, weighted, out)

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
        return _estimate_lockmass_shifts(
            mz_y, self.centroid_func, self.mz_indices, self.peak_indices, self.masks, weighted, out
        )

    @staticmethod
    def apply(mz_y: np.ndarray, shift: int) -> np.ndarray:
        """Apply lockmass shift to the given spectrum.

        Parameters
        ----------
        mz_y : np.ndarray
            Intensity values of the spectrum. This should be a profile mass spectrum with the same number
            of m/z values as used during initialization.
        shift: int
            Shift value to apply.
        """
        return fast_roll(mz_y, shift)

    def estimate_for_reader(self, reader: BaseReader, weighted: bool = True) -> np.ndarray:
        """Estimate lockmass shifts for all spectra in the given reader.

        Parameters
        ----------
        reader : BaseReader
            Reader object to estimate lockmass shifts for. The reader should have a `n_spectra` attribute
            and support indexing to get individual spectra.
        weighted: bool
            Whether to use weighted distance for peak selection.
        """
        if not isinstance(reader, BaseReader):
            raise ValueError("reader must be an instance of BaseReader")
        n_spectra = reader.n_pixels
        shifts = np.zeros((n_spectra, self.n_peaks), dtype=np.float32)
        for i, (_mz_x, mz_y) in enumerate(reader.spectra_iter(silent=False)):
            self.estimate(mz_y, weighted=weighted, out=shifts[i])
        return shifts


def _prepare_lockmass(
    x: np.ndarray, peaks: ty.Iterable[float], window: float = 0.1
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    mz_indices = np.arange(x.shape[0])

    peak_indices = find_nearest_index(x, peaks)
    masks = {peak: get_array_mask(x, peak - window, peak + window) for peak in peaks}
    return mz_indices, peak_indices, masks


def _estimate_lockmass_shifts(
    mz_y: np.ndarray,
    centroid_func: ty.Callable,
    mz_indices: np.ndarray,
    peak_indices: np.ndarray,
    masks: dict,
    weighted: bool = False,
    out: np.ndarray | None = None,
    threshold: float = LOCKMASS_THRESHOLD,
    individual_frac: float = 0.5,
) -> np.ndarray:
    out = np.zeros(peak_indices.size, dtype=np.float32) if out is None else out
    individual_threshold = individual_frac * threshold

    # first, check whether the spectrum has enough intensities
    intensities = [np.max(mz_y[mask]) for mask in masks.values()]
    if np.max(intensities) < threshold:
        return out

    for j, (_peak, mask) in enumerate(masks.items()):
        if intensities[j] < individual_threshold:
            continue
        index = peak_indices[j]
        # get mass spectrum subset
        y = mz_y[mask]
        # generate centroid from the subset spectrum
        cx, cy = centroid_func(mz_indices[mask], y, np.mean(y) * 0.25)

        # if the centroid is empty, skip
        n = len(cx)
        if n == 0:
            continue
        # if the number of centroids is too high, take the top 10
        if n > 10:
            select = np.argsort(cy)[-10:]
            cx, cy = cx[select], cy[select]
        if weighted:
            sort = np.lexsort((cy, np.abs(cx - index)))
        else:
            sort = np.argsort(np.abs(cx - index))
        # calculate actual shift value for particular peak
        out[j] = cx[sort][0] - index
    return out


def fast_roll(array: np.ndarray, num: int, fill_value: int | float = 0) -> np.ndarray:
    """Shift 1d array to a new position with 0 padding to prevent wraparound - this function is actually
    quicker than np.roll.

    Parameters
    ----------
    array : np.ndarray
        array to be shifted
    num : int
        value by which the array should be shifted
    fill_value : Union[float, int]
        value to fill in the areas where wraparound would have happened
    """
    if num == 0:
        return array

    result = np.empty_like(array)
    if not isinstance(num, int):
        raise ValueError("`num` must be an integer")

    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    return result
