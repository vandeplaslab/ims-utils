"""Spectra."""
import typing as ty
from contextlib import suppress

import numba
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def has_pyopenms() -> bool:
    """Check if PyOpenMS is installed."""
    try:
        import pyopenms
    except ImportError:
        return False
    return True


def oms_centroid(mzs: np.ndarray, intensities: np.ndarray, snr: float = 0) -> ty.Tuple[np.ndarray, np.ndarray]:
    """PyOpenMS based centroiding."""
    import pyopenms as oms  # type: ignore

    s, c = oms.MSSpectrum(), oms.MSSpectrum()
    s.set_peaks((mzs, intensities))

    p = oms.PeakPickerHiRes()
    param = p.getDefaults()
    param.update({b"signal_to_noise": float(snr)})
    p.setParameters(param)
    p.pick(s, c)
    cx, cy = c.get_peaks()
    return cx, cy


def picked_centroid(
    mz_array: np.ndarray, int_array: np.ndarray, threshold: float = 0.0
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Convert profile mass spectrum to centroids."""
    snr = 0 if threshold == 0 else np.max(int_array) * threshold
    return oms_centroid(mz_array, int_array, snr)


def maxima_centroid(
    mz_array: np.ndarray, int_array: np.ndarray, weighted_bins: int = 1, min_intensity: float = 1e-5
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Convert profile mass spectrum to centroids."""
    assert len(mz_array) == len(int_array), "Make sure x- and y-axis are of the same size"
    assert weighted_bins < len(mz_array) / 2.0

    mz_array = np.asarray(mz_array).astype(np.float64)
    int_array = np.asarray(int_array).astype(np.float32)

    # calc first & second differential
    gradient_1 = np.gradient(int_array)
    gradient_2 = np.gradient(gradient_1)[0:-1]

    # detect crossing points
    crossing_points = gradient_1[0:-1] * gradient_1[1:] <= 0
    middle_points = gradient_2 < 0

    # check left and right crossing points
    indices_list_l = np.where(crossing_points & middle_points)[0]
    indices_list_r = indices_list_l + 1
    mask = np.where(int_array[indices_list_l] > int_array[indices_list_r], indices_list_l, indices_list_r)
    mask = np.unique(mask)

    # remove peaks below minimum intensities
    mask = mask[int_array[mask] > min_intensity]
    centroid_x = mz_array[mask]
    centroid_y = int_array[mask]

    if weighted_bins > 0:
        # check no peaks within bin width of spectrum edge
        good_idx = (mask > weighted_bins) & (mask < (len(mz_array) - weighted_bins))
        centroid_x = centroid_x[good_idx]
        mask = mask[good_idx]
        r = find_maximum(mz_array, int_array, centroid_x, mask, weighted_bins)
        centroid_x = r[0, :]
        centroid_y = r[1, :]
    return centroid_x, centroid_y


def parabolic_centroid(
    mzs: np.ndarray, intensities: np.ndarray, peak_threshold: float = 0
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Calculate centroid position.

    This function was taken from msiwarp package available on GitHub
    """
    peak_indices, _ = find_peaks(intensities, height=peak_threshold)
    peak_left = peak_indices - 1
    peak_right = peak_indices + 1

    n = len(peak_indices)

    x = np.zeros((n, 3))
    y = np.zeros((n, 3))

    x[:, 0] = mzs[peak_left]
    x[:, 1] = mzs[peak_indices]
    x[:, 2] = mzs[peak_right]

    y[:, 0] = intensities[peak_left]
    y[:, 1] = intensities[peak_indices]
    y[:, 2] = intensities[peak_right]

    a = ((y[:, 2] - y[:, 1]) / (x[:, 2] - x[:, 1]) - (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0])) / (x[:, 2] - x[:, 0])

    b = (
        (y[:, 2] - y[:, 1]) / (x[:, 2] - x[:, 1]) * (x[:, 1] - x[:, 0])
        + (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0]) * (x[:, 2] - x[:, 1])
    ) / (x[:, 2] - x[:, 0])

    mzs_parabolic = (1 / 2) * (-b + 2 * a * x[:, 1]) / a
    intensities_parabolic = a * (mzs_parabolic - x[:, 1]) ** 2 + b * (mzs_parabolic - x[:, 1]) + y[:, 1]
    mask = ~np.isnan(mzs_parabolic)
    return mzs_parabolic[mask], intensities_parabolic[mask]


@numba.njit(fastmath=True, cache=True)
def find_maximum(
    mz_array: np.ndarray, int_array: np.ndarray, centroid_x: np.ndarray, mask: np.ndarray, weighted_bins: int
):
    """Find maxima."""
    result = np.zeros((3, len(centroid_x)))

    for ii in range(len(centroid_x)):
        s = w = 0.0
        max_intensity_idx = 0
        max_intensity = -1
        for k in range(-weighted_bins, weighted_bins + 1):
            idx = mask[ii] + k
            mz = mz_array[idx]
            intensity = int_array[idx]
            w += intensity
            s += mz * intensity
            if intensity > max_intensity:
                max_intensity = intensity
                max_intensity_idx = idx
        result[0][ii] = s / w
        result[1][ii] = max_intensity
        result[2][ii] = max_intensity_idx
    return np.asarray(result)


def resample_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray):
    """Resample array at specified ppm."""
    mz_idx = np.digitize(mz_array, new_mz, True)

    # sum together values from multiple bins
    y_ppm = np.zeros_like(new_mz)
    for i, idx in enumerate(mz_idx):
        with suppress(IndexError):
            y_ppm[idx] += intensity_array[i]
    return y_ppm


def interpolate_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray):
    """Resample array at specified ppm."""
    func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
    return new_mz, func(new_mz)
