"""Detection utilities."""

from __future__ import annotations

import math
import typing as ty

import numpy as np
from koyo.timer import MeasureTimer


def noop_logger(msg: str) -> None:
    """No operation logger."""


def get_best_snr(data: list[tuple[int, float, int]]) -> tuple[int, float, int]:
    """Calculate the best SNR using the Knee method.

    Expects data as tuples of:

        (snr, something, score)

    The knee is found using snr as x and score as y. If no knee can be found,
    the function falls back to the row with the highest score.
    """
    from ims_utils._vendored.kneed import KneeLocator

    res = np.asarray(data, dtype=float)

    if res.ndim != 2 or res.shape[0] == 0 or res.shape[1] < 3:
        raise ValueError("data must be a non-empty list of tuples with at least 3 values")

    x = res[:, 0]
    y = res[:, 2]

    # Remove trailing plateau, but never remove all points.
    max_value = np.max(y)

    keep_until = len(res)
    while keep_until > 1 and y[keep_until - 1] == max_value:
        keep_until -= 1

    # If removing the plateau would leave too few points, keep the original data.
    if keep_until >= 2:
        res_trimmed = res[:keep_until]
    else:
        res_trimmed = res

    x_trimmed = res_trimmed[:, 0]
    y_trimmed = res_trimmed[:, 2]

    # KneeLocator needs at least a few points to behave sensibly.
    if len(res_trimmed) < 3:
        best_idx = int(np.argmax(y))
        return data[best_idx]

    knee = KneeLocator(
        x_trimmed,
        y_trimmed,
        curve_nature="convex",
        curve_direction="increasing",
    )

    if knee.knee is None:
        best_idx = int(np.argmax(y))
        return data[best_idx]

    # knee.knee is an x-value, not necessarily an array/list index.
    best_idx = int(np.argmin(np.abs(x - knee.knee)))

    return data[best_idx]


def decreasing_step_size(
    initial_step: float, iterations: ty.Iterable[int], multiplier: float = 0.05
) -> ty.Generator[float, None, None]:
    """Decreasing step size."""
    for i in iterations:
        step = initial_step * math.exp(-multiplier * i)  # Exponential decay function
        yield step


def optimize_oms_centroid(
    mzs: np.ndarray,
    intensities: np.ndarray,
    initial: float = 0.05,
    iterations: ty.Iterable[int] | None = None,
    multiplier: float = 0.005,
    logger: ty.Callable | None = noop_logger,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Automatically determine appropriate SNR using OpenMS methodology."""
    if logger is None:
        logger = noop_logger
    if iterations is None:
        iterations = np.arange(0, 1000, 25)

    with MeasureTimer() as timer:
        steps = list(decreasing_step_size(initial, iterations, multiplier))
        peaks = oms_centroid_batch(mzs, intensities, steps)

        res = []  # index, snr, n_peaks
        for i, (snr, (px, _py)) in enumerate(peaks.items()):
            res.append((i, snr, len(px)))

        try:
            _, best_snr, _ = get_best_snr(res)
        except IndexError:
            best_snr = 0
        px, py = peaks[best_snr]
    logger(f"Best SNR: {best_snr:.4f} - Found {len(px):,} peaks in {timer()}.")
    return float(best_snr), px, py


def optimize_oms_centroid_by_min_peaks(
    mzs: np.ndarray, intensities: np.ndarray, min_peaks: int, initial: float = 0.05, step_size: float = 0.0005
) -> tuple[float, np.ndarray, np.ndarray]:
    """Automatically determine appropriate SNR using OpenMS methodology."""
    from pyopenms import MSSpectrum, PeakPickerHiRes

    s, c = MSSpectrum(), MSSpectrum()
    s.set_peaks((mzs, intensities))

    while True:
        p = PeakPickerHiRes()
        param = p.getDefaults()
        param.update({b"signal_to_noise": float(initial)})
        p.setParameters(param)
        p.pick(s, c)
        cx, cy = c.get_peaks()
        if len(cx) >= min_peaks:
            break
        initial -= step_size
        if initial <= 0:
            break
    return float(initial), cx, cy


def oms_centroid_batch(mzs: np.ndarray, intensities: np.ndarray, snr: ty.Iterable[float] = ("0",)):
    """Centroids using OpenMS methodology in batch."""
    from pyopenms import MSSpectrum, PeakPickerHiRes

    s, c = MSSpectrum(), MSSpectrum()
    s.set_peaks((mzs, intensities))

    res = {}
    for snr_ in snr:
        p = PeakPickerHiRes()
        param = p.getDefaults()
        param.update({b"signal_to_noise": float(snr_)})
        p.setParameters(param)
        p.pick(s, c)
        cx, cy = c.get_peaks()
        res[snr_] = (cx, cy)
    return res
