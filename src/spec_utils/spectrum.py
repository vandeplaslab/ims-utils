"""Spectra."""

from __future__ import annotations

import math
import typing as ty
from contextlib import suppress

import numba
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import decimate, find_peaks, resample, resample_poly, upfirdn


def has_pyopenms() -> bool:
    """Check if PyOpenMS is installed."""
    try:
        import pyopenms
    except ImportError:
        return False
    return True


def centwave_estimate_baseline_and_noise(y: np.ndarray):
    """Calculate signal-to-noise ratio."""
    q_min, q_max = np.quantile(y, (0.05, 0.95))
    indices = np.where((y > q_min) & (y < q_max))
    yt = y[indices]
    bl, nl = np.mean(yt), np.std(yt)
    return bl, nl


def centwave_estimate_noise(y: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculate signal-to-noise ratio."""
    bl, nl = centwave_estimate_baseline_and_noise(y)
    return (ys - bl) / nl


def oms_centroid(
    mzs: np.ndarray,
    intensities: np.ndarray,
    snr: float = 0,
    spacing_difference: float = 1.5,
    spacing_difference_gap: float = 4.0,
    snr_auto_mode: int = 0,
    **_kwargs,
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """PyOpenMS based centroiding."""
    import pyopenms as oms  # type: ignore

    s, c = oms.MSSpectrum(), oms.MSSpectrum()
    s.set_peaks((mzs, intensities))

    p = oms.PeakPickerHiRes()
    param = p.getDefaults()
    param.update(
        {
            b"signal_to_noise": float(snr),
            b"spacing_difference": float(spacing_difference),
            b"spacing_difference_gap": float(spacing_difference_gap),
            b"SignalToNoise:auto_mode": int(snr_auto_mode),
            b"SignalToNoise:write_log_messages": b"false",
        }
    )
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


def downsample(array_signal, n_up=2, n_down=20, ds_filter="upfirdn"):
    """Downsample signal using one of multiple methods.

    Parameters
    ----------
    array_signal : np.array, shape=(n_dt)
        array to be downsampled
    n_up : int
        upsampling rate
    n_down : int
        downsampling rate
    ds_filter : {'fir', 'upfirdn', 'decimate', 'poly', 'polynomial', 'fft'}
        downsampling filter

    Returns
    -------
    downsampled_signal : np.array
        downsampled signal
    """
    ratio = n_down / n_up
    out_size = math.ceil(array_signal.shape[0] / ratio)
    if ds_filter in ["fir", "upfirdn"]:
        downsampled_signal = upfirdn([1], array_signal, n_up, n_down)
    elif ds_filter == "decimate":
        downsampled_signal = decimate(array_signal, math.floor(ratio), ftype="fir")
    elif ds_filter in ["poly", "polynomial"]:
        downsampled_signal = resample_poly(array_signal, n_up, n_down)
    elif ds_filter == "fft":
        downsampled_signal = resample(array_signal, out_size)
    else:
        downsampled_signal = array_signal
    return downsampled_signal


def merge_peaks_by_ppm(
    mz: np.ndarray, intensity: np.ndarray, ion_mobility: np.ndarray | None = None, combine_ppm: float = 5
):
    """Merge peaks by ppm."""
    from koyo.spectrum import ppm_diff

    if ion_mobility is None:
        ion_mobility = np.zeros_like(mz)

    # this calculates the ppm difference between adjacent m/z values and subsequently splits those masses
    # into separate groups
    splits = np.where(ppm_diff(mz) > combine_ppm)[0] + 1
    index = np.asarray([group.argmax() for group in np.split(intensity, splits)])
    x, y, im = [], [], []
    for i, (group_x, group_y, group_im) in enumerate(
        zip(
            np.split(mz, splits),
            np.split(intensity, splits),
            np.split(ion_mobility, splits),
        )
    ):
        x.append(group_x[index[i]])
        y.append(group_y[index[i]])
        im.append(group_im[index[i]])
    return np.asarray(x), np.asarray(y), np.asarray(im)


def merge_peaks_by_tol(
    mz: np.ndarray, intensity: np.ndarray, ion_mobility: np.ndarray | None = None, combine_tol: float = 0.001
):
    """Merge peaks by m/z tolerance."""
    if ion_mobility is None:
        ion_mobility = np.zeros_like(mz)
    index_groups, peak_groups = group_peaks_by_tol(mz, combine_tol)
    mz, intensity, im = apply_most_intense(mz, intensity, index_groups, ion_mobility)
    return mz, intensity, im


def select_most_intense(xs, ys, index_groups):
    """Select peaks that have the highest intensity."""
    xs_new, ys_new, indices_sel = [], [], []
    for index_group in index_groups:
        indices = np.argsort(ys[index_group])
        xs_new.append(xs[index_group][indices[-1]])
        ys_new.append(ys[index_group][indices[-1]])
        indices_sel.append(index_group[indices[-1]])
    return np.array(xs_new), np.array(ys_new)


def get_most_intense_index(xs, ys, index_groups):
    """Get most intense index."""
    indices_sel = []
    for index_group in index_groups:
        indices = np.argsort(ys[index_group])
        indices_sel.append(index_group[indices[-1]])
    return indices_sel


def apply_most_intense(xs, ys, index_groups, *arrays):
    """Apply most intense."""
    indices_sel = get_most_intense_index(xs, ys, index_groups)
    ret = []
    for _ in range(len(arrays) + 2):
        ret.append([])
    for index in indices_sel:
        ret[0].append(xs[index])
        ret[1].append(ys[index])
        for i, array in enumerate(arrays, start=2):
            ret[i].append(array[index])
    return [np.array(r) for r in ret]


def merge_peaks(xs: ty.List[np.ndarray], ys: ty.List[np.ndarray], threshold: float = 0.005):
    """Merge peaks."""
    xs, indices = np.unique(np.concatenate(xs), return_index=True)
    ys = np.concatenate(ys)[indices]
    index_groups, peak_groups = group_peaks_by_tol(xs, threshold)
    xs_new, ys_new = select_most_intense(xs, ys, index_groups)
    return xs_new, ys_new


def group_peaks_by_tol(xs: np.ndarray, threshold: float = 0.01) -> ty.Tuple[ty.List[np.ndarray], ty.List[np.ndarray]]:
    """
    Group peaks together by their distance based on a threshold.

    Parameters
    ----------
    xs : np.ndarray
        Sorted array of m/z values.
    threshold : float
        Maximum difference between peaks for grouping.

    Returns
    -------
        List of Lists: A list of peak groups, where each group is represented as a list of m/z values.
    """
    index_groups = []
    current_group = []

    for i in range(len(xs) - 1):
        current_group.append(i)

        # Check if the next peak is within the threshold
        if xs[i + 1] - xs[i] > threshold:
            index_groups.append(current_group)
            current_group = []

    # Add the last group
    if len(xs) > 0:
        current_group.append(len(xs) - 1)
        index_groups.append(current_group)
    index_groups = [np.array(index_group) for index_group in index_groups]
    peak_groups = [xs[index_group] for index_group in index_groups]
    return index_groups, peak_groups
