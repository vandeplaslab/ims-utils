"""Spectra."""

from __future__ import annotations

import math
import typing as ty
import warnings
from bisect import bisect_left, bisect_right
from contextlib import suppress

import numba
import numpy as np
from koyo.typing import SimpleArrayLike
from koyo.utilities import find_nearest_index, find_nearest_index_batch
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import decimate, find_peaks, resample, resample_poly, upfirdn
from scipy.signal._peak_finding import _select_by_property, _unpack_condition_args


def has_pyopenms() -> bool:
    """Check if PyOpenMS is installed."""
    try:
        import pyopenms
    except ImportError:
        return False
    return True


def centwave_estimate_baseline_and_noise(y: np.ndarray) -> tuple[float, float]:
    """Calculate signal-to-noise ratio."""
    q_min, q_max = np.quantile(y, (0.05, 0.95))
    if q_min == q_max == 0:
        q_min, q_max = np.quantile(y[y > 0], (0.05, 0.95))
    indices = np.where((y > q_min) & (y < q_max))
    yt = y[indices]
    bl, nl = np.mean(yt), np.std(yt)
    return bl, nl


def centwave_estimate_noise(y: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculate signal-to-noise ratio."""
    bl, nl = centwave_estimate_baseline_and_noise(y)
    return (ys - bl) / nl



def local_estimate_noise(
    mz: np.ndarray,
    intens: np.ndarray,
    peaks: ty.Union[ty.Sequence[int], ty.Sequence[float]],
    *,
    peaks_kind: ty.Literal["index", "mz"] = "index",
    window_ppm: float = 50.0,
    window_da: float = None,
    core_ppm: float = 5.0,
    core_da: float = None,
    min_points: int = 25,
    clip_nonpositive_noise: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate SNR for peaks in a mass spectrum using robust local baseline & noise.

    Parameters
    ----------
    mz : (N,) array
        Monotonic m/z axis (profile or centroid grid).
    intens : (N,) array
        Intensities aligned with `mz`.
    peaks : sequence of ints or floats
        Peak locations. If peaks_kind="index", they are indices into the arrays.
        If peaks_kind="mz", they are target m/z values (nearest index is used).
    peaks_kind : {'index','mz'}
        How to interpret `peaks`.
    window_ppm, window_da :
        Half-width of the *background* window around each peak. Use ppm (instrument-aware)
        or absolute Da (if provided). If both provided, the **max** of the two is used.
    core_ppm, core_da :
        Half-width of the *core* exclusion zone centered on the peak that is removed
        from the background window before estimating baseline/noise. Again, max of ppm/Da.
        Typical defaults: core_ppm ~ 3–10 ppm for high-res; core_da ~ 0.001–0.01 Da.
    min_points : int
        Minimum number of points required in the background set; if fewer, the window is
        expanded 1.5× iteratively until satisfied (bounded by spectrum edges).
    clip_nonpositive_noise : float
        Floor to avoid division by zero.

    Returns
    -------
    snr : (K,) array
        SNR for each input peak.
    baseline : (K,) array
        Estimated local baseline (median of background).
    noise_sigma : (K,) array
        Robust noise sigma estimate = 1.4826 * MAD(background - baseline).

    Notes
    -----
    - Background is the set of samples within `window` but **outside** the `core`.
    - Peak "height" is the intensity at the peak index (no sub-bin refinement here).
    - For centroided data with sparse points, you can increase `window_ppm` so the
      background includes enough samples.
    """
    mz = np.asarray(mz, dtype=float)
    y = np.asarray(intens, dtype=float)
    n = mz.size
    if y.size != n:
        raise ValueError("mz and intens must have the same length.")

    # Resolve peak indices
    if peaks_kind == "index":
        idx = np.asarray(peaks, dtype=int)
        if np.any((idx < 0) | (idx >= n)):
            raise ValueError("Some peak indices are out of range.")
    elif peaks_kind == "mz":
        pmz = np.asarray(peaks, dtype=float)
        # nearest neighbor indices via searchsorted
        idx = np.clip(np.searchsorted(mz, pmz), 0, n-1)
        # choose closer neighbor if we're on the right side
        right = idx < n-1
        choose_left = right & (np.abs(pmz[right] - mz[idx[right]]) > np.abs(pmz[right] - mz[idx[right]-1]))
        idx[right][choose_left] -= 1
    else:
        raise ValueError("peaks_kind must be 'index' or 'mz'.")

    # Helpers to turn ppm/Da into absolute Da windows at a given mz
    def _halfwidth_da(m0: float, ppm: float, da: float) -> float:
        w_ppm = m0 * ppm * 1e-6 if ppm is not None else 0.0
        w_da = da if da is not None else 0.0
        return max(w_ppm, w_da, 0.0)

    snr = np.empty(idx.size, dtype=float)
    baseline = np.empty(idx.size, dtype=float)
    noise_sig = np.empty(idx.size, dtype=float)

    for k, i in enumerate(idx):
        m0 = mz[i]

        # Determine window and core half-widths in Da
        w_da = _halfwidth_da(m0, window_ppm, window_da)
        c_da = _halfwidth_da(m0, core_ppm, core_da)

        # Ensure the core is not larger than the window
        if c_da >= w_da:
            # fall back to a smaller core (e.g., 1/3 of window)
            c_da = 0.33 * w_da

        # Convert Da to index ranges using searchsorted
        def range_idx(da_half: float):
            lo = m0 - da_half
            hi = m0 + da_half
            a = np.searchsorted(mz, lo, side="left")
            b = np.searchsorted(mz, hi, side="right")
            return max(0, a), min(n, b)

        a_w, b_w = range_idx(w_da)
        a_c, b_c = range_idx(c_da)

        # Background = window minus core slice
        # Concatenate left and right sides (avoid copying huge arrays by views)
        # If too few points, scale window until enough or edges hit
        scale = 1.0
        bg = None
        while True:
            if scale != 1.0:
                a_w, b_w = range_idx(scale * w_da)
                a_c, b_c = range_idx(scale * c_da * 0.8)  # keep core smaller than window
            left_bg = y[a_w:max(a_c, a_w)]
            right_bg = y[min(b_c, b_w):b_w]
            if left_bg.size + right_bg.size >= min_points or (a_w == 0 and b_w == n):
                bg = (left_bg, right_bg)
                break
            scale *= 1.5
            if scale > 20:  # safety
                bg = (left_bg, right_bg)
                break

        # Robust baseline & noise from background
        if left_bg.size + right_bg.size == 0:
            # Degenerate case: no background, fall back to global estimates
            med = np.median(y)
            mad = np.median(np.abs(y - med))
        else:
            # Use medians without concatenating to save memory
            comb = np.concatenate(bg) if left_bg.size and right_bg.size else (left_bg if left_bg.size else right_bg)
            med = np.median(comb)
            mad = np.median(np.abs(comb - med))

        sigma = 1.4826 * mad
        sigma = max(sigma, clip_nonpositive_noise)

        baseline[k] = med
        noise_sig[k] = sigma
        snr[k] = (y[i] - med) / sigma
    return snr, baseline, noise_sig



def estimate_peak_prominence(
    intensities: np.ndarray,
    peaks: ty.Union[ty.Sequence[int], np.ndarray],
    *,
    smooth_median_width: int = 0,
) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak prominence for a 1D signal (mass spectrum).

    Parameters
    ----------
    intensities : (N,) array
        Intensities (profile or centroid-grid). Values should be non-negative.
    peaks : sequence of int
        Indices of peaks (e.g., from your peak picker, local maxima).
    smooth_median_width : int, default 0
        Optional simple denoising: if >0, apply a running median of this width
        *only* for determining valley minima (the returned prominences are still
        referenced to original intensities). Use an odd integer (e.g., 5–21).

    Returns
    -------
    prominence : (K,) array
        Prominence of each peak.
    left_base : (K,) array of int
        Index of the left base (valley) used for the prominence contour.
    right_base : (K,) array of int
        Index of the right base (valley) used for the prominence contour.

    Notes
    -----
    - This mirrors the standard definition used in signal processing libraries.
    - Complexity is O(K log K + K * avg_valley_span); fast in practice since K << N.
    - If a peak has no higher neighbor on one side, that side’s valley search extends
      to the array edge.
    """
    y = np.asarray(intensities, dtype=float)
    n = y.size
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
    if np.any(peaks < 0) or np.any(peaks >= n):
        raise ValueError("Some peak indices are out of range.")

    # Optional mild smoothing (for valley-finding only)
    if smooth_median_width and smooth_median_width > 1:
        k = int(smooth_median_width)
        k = k if k % 2 == 1 else k + 1
        pad = k // 2
        pad_left = y[0] * np.ones(pad)
        pad_right = y[-1] * np.ones(pad)
        yp = np.concatenate([pad_left, y, pad_right])
        # running median via sliding window (simple but OK for valley-finding)
        ys = np.empty_like(y)
        for i in range(n):
            ys[i] = np.median(yp[i:i + 2*pad + 1])
        y_valley = ys
    else:
        y_valley = y  # use raw for valley search

    # Sort peaks by index
    order = np.argsort(peaks)
    p_idx = peaks[order]
    p_h = y[p_idx]

    # Nearest higher peak to LEFT and RIGHT among the given peaks
    # Standard "nearest greater element" using stacks.
    left_higher = np.full(p_idx.size, -1, dtype=int)
    stack = []  # will store indices into p_idx (monotone decreasing heights)
    for j in range(p_idx.size):
        h = p_h[j]
        # Maintain stack of strictly greater heights
        while stack and p_h[stack[-1]] <= h:
            stack.pop()
        left_higher[j] = stack[-1] if stack else -1
        stack.append(j)

    right_higher = np.full(p_idx.size, -1, dtype=int)
    stack = []
    for j in range(p_idx.size - 1, -1, -1):
        h = p_h[j]
        while stack and p_h[stack[-1]] <= h:
            stack.pop()
        right_higher[j] = stack[-1] if stack else -1
        stack.append(j)

    # For each peak, define the search bounds to the left/right
    left_base = np.empty(p_idx.size, dtype=int)
    right_base = np.empty(p_idx.size, dtype=int)
    prominence = np.empty(p_idx.size, dtype=float)

    for j in range(p_idx.size):
        i = p_idx[j]
        # Left bound: index after the nearest higher peak on the left (or start of signal)
        if left_higher[j] == -1:
            left_bound = 0
        else:
            left_bound = p_idx[left_higher[j]]
        # Right bound: index before the nearest higher peak on the right (or end of signal)
        if right_higher[j] == -1:
            right_bound = n - 1
        else:
            right_bound = p_idx[right_higher[j]]

        # Left valley: minimum between (left_bound .. i)
        if i > left_bound:
            lb_slice = slice(left_bound, i)
            il = left_bound + int(np.argmin(y_valley[lb_slice]))
        else:
            il = i
        # Right valley: minimum between (i .. right_bound)
        if right_bound > i:
            rb_slice = slice(i, right_bound + 1)
            ir = i + int(np.argmin(y_valley[rb_slice]))
        else:
            ir = i

        # Contour level is the higher of the two valleys
        base_level = max(y_valley[il], y_valley[ir])
        prom = y[i] - base_level
        if prom < 0:
            prom = 0.0  # guard tiny negatives due to noise/smoothing differences

        left_base[j] = il
        right_base[j] = ir
        prominence[j] = prom

    # Return in the same order as input peaks
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return prominence[inv], left_base[inv], right_base[inv]


def oms_centroid(
    mzs: np.ndarray,
    intensities: np.ndarray,
    snr: float = 0,
    spacing_difference: float = 1.5,
    spacing_difference_gap: float = 4.0,
    snr_auto_mode: int = 0,
    **_kwargs: ty.Any,
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Convert profile mass spectrum to centroids."""
    snr = 0 if threshold == 0 else np.max(int_array) * threshold
    return oms_centroid(mz_array, int_array, snr)


def maxima_centroid(
    mz_array: np.ndarray, int_array: np.ndarray, weighted_bins: int = 1, min_intensity: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate centroid position.

    This function was taken from msiwarp package available on GitHub
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak_indices, _ = find_peaks(intensities, height=peak_threshold)
        return _parabolic_centroid(mzs, intensities, peak_indices)


def fast_parabolic_centroid(
    mzs: np.ndarray, intensities: np.ndarray, peak_threshold: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate centroid position.

    This function was taken from msiwarp package available on GitHub
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak_indices = fast_find_peaks(intensities, height=peak_threshold)
        return _parabolic_centroid(mzs, intensities, peak_indices)


def _parabolic_centroid(
    mzs: np.ndarray, intensities: np.ndarray, peak_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def fast_find_peaks(y: np.ndarray, height: float = 0) -> np.ndarray:
    """Faster implementation of find_peaks without all the bells and whistles."""
    peaks, left_edges, right_edges = _local_maxima_1d(y)

    if height is not None and height > 0:
        peak_heights = y[peaks]
        hmin, hmax = _unpack_condition_args(height, y, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
    return peaks  # type: ignore[no-any-return]


@numba.njit(fastmath=True, cache=True)  # type: ignore[misc]
def _local_maxima_1d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find local maxima in a 1D array.

    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.

    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    left_edges : ndarray
        Indices of edges to the left of local maxima in `x`.
    right_edges : ndarray
        Indices of edges to the right of local maxima in `x`.
    """
    n = x.shape[0]
    max_possible = n // 2

    midpoints = np.empty(max_possible, dtype=np.intp)
    left_edges = np.empty(max_possible, dtype=np.intp)
    right_edges = np.empty(max_possible, dtype=np.intp)
    m = 0  # Number of maxima found

    i = 1
    i_max = n - 1

    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1

            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1

            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                i = i_ahead
                continue  # Skip next increment since we already advanced
        i += 1
    return midpoints[:m], left_edges[:m], right_edges[:m]


@numba.njit(fastmath=True, cache=True)  # type: ignore[misc]
def find_maximum(
    mz_array: np.ndarray, int_array: np.ndarray, centroid_x: np.ndarray, mask: np.ndarray, weighted_bins: int
) -> np.ndarray:
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


def resample_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray) -> np.ndarray:
    """Resample array at specified ppm."""
    mz_idx = np.digitize(mz_array, new_mz, True)

    # sum together values from multiple bins
    y_ppm = np.zeros_like(new_mz)
    for i, idx in enumerate(mz_idx):
        with suppress(IndexError):
            y_ppm[idx] += intensity_array[i]
    return y_ppm


def interpolate_ppm(
    new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample array at specified ppm."""
    func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
    return new_mz, func(new_mz)


def downsample(
    array: np.ndarray,
    n_up: int = 2,
    n_down: int = 20,
    ds_filter: ty.Literal["fir", "upfirdn", "decimate", "poly", "polynomial", "fft"] = "upfirdn",
) -> np.ndarray:
    """Downsample signal using one of multiple methods.

    Parameters
    ----------
    array : np.array, shape=(n_dt)
        array to be downsampled
    n_up : int
        upsampling rate
    n_down : int
        downsampling rate
    ds_filter : {'fir', 'upfirdn', 'decimate', 'poly', 'polynomial', 'fft'}
        downsampling filter

    Returns
    -------
    ds_array : np.array
        downsampled signal
    """
    ratio = n_down / n_up
    out_size = math.ceil(array.shape[0] / ratio)
    if ds_filter in ["fir", "upfirdn"]:
        ds_array = upfirdn([1], array, n_up, n_down)
    elif ds_filter == "decimate":
        ds_array = decimate(array, math.floor(ratio), ftype="fir")
    elif ds_filter in ["poly", "polynomial"]:
        ds_array = resample_poly(array, n_up, n_down)
    elif ds_filter == "fft":
        ds_array = resample(array, out_size)
    else:
        ds_array = array
    return ds_array


def merge_peaks_by_ppm(
    mz: np.ndarray, intensity: np.ndarray, ion_mobility: np.ndarray | None = None, combine_ppm: float = 5
):
    """Merge peaks by ppm."""
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


def select_most_intense(xs: np.ndarray, ys: np.ndarray, index_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select peaks that have the highest intensity."""
    xs_new, ys_new, indices_sel = [], [], []
    for index_group in index_groups:
        indices = np.argsort(ys[index_group])
        xs_new.append(xs[index_group][indices[-1]])
        ys_new.append(ys[index_group][indices[-1]])
        indices_sel.append(index_group[indices[-1]])
    return np.array(xs_new), np.array(ys_new)


def get_most_intense_index(ys: np.ndarray, index_groups: np.ndarray) -> np.ndarray:
    """Get most intense index."""
    indices_sel = []
    for index_group in index_groups:
        indices = np.argsort(ys[index_group])
        indices_sel.append(index_group[indices[-1]])
    return np.asarray(indices_sel)


def apply_most_intense(
    xs: np.ndarray, ys: np.ndarray, index_groups: np.ndarray, *arrays: np.ndarray
) -> list[np.ndarray]:
    """Apply most intense."""
    indices_sel = get_most_intense_index(ys, index_groups)
    ret = []
    for _ in range(len(arrays) + 2):
        ret.append([])
    for index in indices_sel:
        ret[0].append(xs[index])
        ret[1].append(ys[index])
        for i, array in enumerate(arrays, start=2):
            ret[i].append(array[index])
    return [np.array(r) for r in ret]


def merge_peaks(xs: list[np.ndarray], ys: list[np.ndarray], threshold: float = 0.005):
    """Merge peaks."""
    xs, indices = np.unique(np.concatenate(xs), return_index=True)
    ys = np.concatenate(ys)[indices]
    index_groups, peak_groups = group_peaks_by_tol(xs, threshold)
    xs_new, ys_new = select_most_intense(xs, ys, index_groups)
    return xs_new, ys_new


def group_peaks_by_tol(xs: np.ndarray, threshold: float = 0.01) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def get_ppm_axis(mz_start: float, mz_end: float, ppm: float):
    """Compute sequence of m/z values at a particular ppm."""
    import math

    if mz_start == 0 or mz_end == 0 or ppm == 0:
        raise ValueError("Input values cannot be equal to 0.")
    # Use the square root to correct the bin spacing
    step_ratio = np.sqrt((1 + 1e-6 * ppm) / (1 - 1e-6 * ppm))

    # Calculate the length using the corrected step ratio
    length = (np.log(mz_end) - np.log(mz_start)) / np.log(step_ratio)
    length = math.floor(length) + 1

    # Calculate m/z values using the corrected step ratio
    mz = mz_start * np.power(step_ratio, np.arange(length))
    return mz


@numba.njit(cache=True, fastmath=True)  # type: ignore[misc]
def find_between(data: SimpleArrayLike, min_value: float, max_value: float):
    """Find indices between windows."""
    return np.where(np.logical_and(data >= min_value, data <= max_value))[0]


@numba.njit(cache=True, fastmath=True)  # type: ignore[misc]
def find_between_tol(data: np.ndarray, value: float, tol: float):
    """Find indices between window and ppm."""
    return find_between(data, value - tol, value + tol)


@numba.njit(cache=True, fastmath=True)  # type: ignore[misc]
def find_between_ppm(data: np.ndarray, value: float, ppm: float):
    """Find indices between window and ppm."""
    window = get_window_for_ppm(value, abs(ppm))
    return find_between(data, value - window, value + window)


@numba.njit(cache=True, fastmath=True)  # type: ignore[misc]
def find_between_batch(array: np.ndarray, min_value: np.ndarray, max_value: np.ndarray):
    """Find indices between specified boundaries for many items."""
    min_indices = np.searchsorted(array, min_value, side="left")
    max_indices = np.searchsorted(array, max_value, side="right")

    res = []
    for i in range(len(min_value)):
        _array = array[min_indices[i] : max_indices[i]]
        res.append(min_indices[i] + find_between(_array, min_value[i], max_value[i]))
    return res


@numba.njit(fastmath=True, cache=True)
def get_window_for_ppm(mz: float, ppm: float) -> float:
    """Calculate window size for specified peak at specified ppm."""
    step = mz * 1e-6  # calculate appropriate step size for specified mz value
    peak_x_ppm = mz
    is_subtract = ppm < 0
    ppm = abs(ppm)
    while True:
        if ((peak_x_ppm - mz) / mz) * 1e6 >= ppm:
            break
        peak_x_ppm += step
    value = peak_x_ppm - mz
    return value if not is_subtract else -value


def ppm_diff(a: np.ndarray, axis=-1) -> np.ndarray:
    """Calculate the ppm difference between set of values in array.

    This function is inspired by `np.diff` which very efficiently computes the difference between adjacent points.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from numpy.core.multiarray import normalize_axis_index

    a = np.asarray(a, dtype=np.float32)
    nd = a.ndim
    axis = normalize_axis_index(axis, nd)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    return (np.subtract(a[slice1], a[slice2]) / a[slice2]) * 1e6


def running_average(x: np.ndarray, size: int) -> np.ndarray:
    """Running average."""
    return uniform_filter1d(x, size, mode="nearest")


def _cluster_within_ppm_with_index(array: np.ndarray, ppm: float):
    """Cluster results within ppm tolerance."""
    tmp = array.copy()
    indices = np.arange(tmp.size)
    groups = []
    index_groups = []
    while len(tmp):
        # select seed
        seed = tmp.min()
        mask = np.abs(ppm_error(tmp, seed)) <= ppm
        groups.append(tmp[mask])
        index_groups.append(indices[mask])
        tmp = tmp[~mask]
        indices = indices[~mask]
    return groups, index_groups


def cluster_within_ppm(array: np.ndarray, ppm: float):
    """Cluster results within ppm tolerance."""
    tmp = array.copy()
    groups = []
    while len(tmp):
        # select seed
        seed = tmp.min()
        mask = np.abs(ppm_error(tmp, seed)) <= ppm
        groups.append(tmp[mask])
        tmp = tmp[~mask]
    return groups


@numba.njit(fastmath=True, cache=True)  # type: ignore[misc]
def ppm_to_delta_mass(mz: float | np.ndarray, ppm: float | np.ndarray) -> float | np.ndarray:
    """Converts a ppm error range to a delta mass in th (da?).

    Parameters
    ----------
    mz : float
        Observed m/z
    ppm : float
        mass range in ppm

    Example
    -------
    ppm_to_delta_mass(1234.567, 50)
    """
    return ppm * mz / 1_000_000.0


@numba.njit(fastmath=True, cache=True)  # type: ignore[misc]
def ppm_error(measured_mz: float | np.ndarray, theoretical_mz: float | np.ndarray) -> float | np.ndarray:
    """Calculate ppm error."""
    return ((measured_mz - theoretical_mz) / theoretical_mz) * 1e6


def select_nearest_index(values: np.ndarray, values_to_find: np.ndarray, ppm: float = 3.0) -> np.ndarray:
    """Find nearest values within a specified window."""
    indices = find_nearest_index(values, values_to_find)
    found_values = values[indices]
    mask = np.abs(ppm_error(found_values, values_to_find)) <= ppm
    return indices[mask]


def select_nearest(values: np.ndarray, values_to_find: np.ndarray, ppm: float = 3.0) -> np.ndarray:
    """Find nearest values within a specified window."""
    indices = select_nearest_index(values, values_to_find, ppm)
    return values[indices]


def get_peaklist_window_for_ppm(peaklist: np.ndarray, ppm: float) -> list[tuple[float, float]]:
    """Retrieve peaklist + tolerance."""
    _peaklist = []
    for mz in peaklist:
        _peaklist.append((mz, get_window_for_ppm(mz, ppm)))
    return _peaklist


def get_peaklist_window_for_da(peaklist: np.ndarray, da: float) -> list[tuple[float, float]]:
    """Retrieve peaklist + tolerance."""
    _peaklist = []
    for mz in peaklist:
        _peaklist.append((mz, da))
    return _peaklist


def get_mzs_for_tol(mzs: np.ndarray, tol: float | None = None, ppm: float | None = None):
    """Get min/max values for specified tolerance or ppm."""
    if (tol is None and ppm is None) or (tol == 0 and ppm == 0):
        raise ValueError("Please specify `tol` or `ppm`.")
    elif tol is not None and ppm is not None:
        raise ValueError("Please only specify `tol` or `ppm`.")

    mzs = np.asarray(mzs)
    if tol:
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    else:
        tol = np.asarray([get_window_for_ppm(mz, ppm) for mz in mzs])
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    return mzs_min, mzs_max


def bisect_spectrum(x_spectrum: np.ndarray, mz_value: np.ndarray, tol: float) -> tuple[int, int]:
    """Get left/right window of extraction for peak."""
    ix_l, ix_u = (
        bisect_left(x_spectrum, mz_value - tol),
        bisect_right(x_spectrum, mz_value + tol) - 1,
    )
    if ix_l == len(x_spectrum):
        return len(x_spectrum), len(x_spectrum)
    if ix_u < 1:
        return 0, 0
    if ix_u == len(x_spectrum):
        ix_u -= 1
    if x_spectrum[ix_l] < (mz_value - tol):
        ix_l += 1
    if x_spectrum[ix_u] > (mz_value + tol):
        ix_u -= 1
    return ix_l, ix_u


@numba.njit()  # type: ignore[misc]
def trim_axis(x: np.ndarray, y: np.ndarray, min_val: float, max_val: float) -> tuple[np.ndarray, np.ndarray]:
    """Trim axis to prevent accumulation of edges."""
    mask = np.where((x >= min_val) & (x <= max_val))
    return x[mask], y[mask]


@numba.njit()  # type: ignore[misc]
def set_ppm_axis(mz_x: np.ndarray, mz_y: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Set values for axis."""
    mz_idx = np.digitize(x, mz_x, True)
    for i, idx in enumerate(mz_idx):
        mz_y[idx] += y[i]
    return mz_y


def get_ppm_offsets(mz_x: np.ndarray, ppm: float, min_spacing: float = 1e-5, every_n: int = 100) -> np.ndarray:
    """Generate correction map of specified ppm."""
    spacing = min_spacing  # if ppm > 0 else -min_spacing
    is_subtract = ppm < 0
    ppm = abs(ppm)
    _mzx = mz_x[::every_n]
    result = np.zeros_like(_mzx)
    mzx = mz_x
    full_result = np.zeros_like(mzx)
    if ppm == 0:
        return full_result

    n = 10
    index_offset = 0
    for i, val in enumerate(_mzx):
        while True:
            offsets = np.full(n, spacing) * np.arange(index_offset, index_offset + n)
            errors = ppm_error(val, val - offsets)
            index = find_nearest_index(errors, ppm)
            nearest = errors[index]
            if nearest >= ppm:
                offset = offsets[index]
                break
            index_offset += n
        result[i] = offset
        index_offset -= n * 2

    start_idx = 0
    indices = find_nearest_index_batch(mzx, _mzx)
    indices[-1] = len(mzx)
    for i, end_idx in enumerate(indices):
        full_result[start_idx:end_idx] = result[i]
        start_idx = end_idx
    return full_result if not is_subtract else -full_result


def get_multi_ppm_offsets(mz_x: np.ndarray, ppm_ranges, min_spacing: float = 1e-5, every_n: int = 100) -> np.ndarray:
    """Generate correction map of specified ppm."""
    start_offset = 0
    _mzx = mz_x[::every_n]
    offsets = np.zeros_like(_mzx)
    mzx = mz_x
    full_offsets = np.zeros_like(mzx)
    if all(_ppm[2] == 0 for _ppm in ppm_ranges):
        return full_offsets

    ppm_ = []
    for x_min, x_max, ppm in ppm_ranges:
        spacing = min_spacing if ppm > 0 else -min_spacing
        ppm_.append((x_min, x_max, ppm, spacing))

    for i, val in enumerate(_mzx):
        offset = start_offset
        for x_min, x_max, ppm, spacing in ppm_:
            if x_min <= val <= x_max:
                if ppm > 0:
                    while ppm_error(val, val - offset) <= ppm:
                        offset += spacing
                else:
                    while ppm_error(val, val - offset) >= ppm:
                        offset += spacing
                break
        offsets[i] = offset

    start_idx = 0
    indices = find_nearest_index_batch(mzx, _mzx)
    indices[-1] = len(mzx)
    for i, end_idx in enumerate(indices):
        full_offsets[start_idx:end_idx] = offsets[i]
        start_idx = end_idx
    return full_offsets


def _precompile() -> None:
    """Precompile numba functions."""
    import os

    if os.environ.get("IMSUTILS_JIT_PRE", "0") == "0":
        return
    for value in [10, np.array([10, 11])]:
        ppm_error(value, value)
    x = np.arange(100)
    y = np.random.random(100)
    fast_parabolic_centroid(x, y)


_precompile()
