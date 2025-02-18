"""Detection utilities."""

from __future__ import annotations

import math
import typing as ty

import numpy as np
from koyo.timer import MeasureTimer


def noop_logger(msg: str) -> None:
    """No operation logger."""


def get_best_snr(data: list[tuple[int, float, int]]) -> tuple[int, float, int]:
    """Calculate best SNR using the Knee method often used in K-Means clustering."""
    from ims_utils._vendored.kneed import KneeLocator

    # convert data to numpy array
    res = np.asarray(data)
    # check if there is a plateau at the end and if so, let's remove the last plateau
    max_value = res[:, 2].max()
    while res[-1, 2] == max_value:
        res = res[:-1]

    # using knee locator to find the best SNR
    knee = KneeLocator(res[:, 0], res[:, 2], curve_nature="convex", curve_direction="increasing")
    best = int(knee.knee)
    return data[best]


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

        best = get_best_snr(res)
        _, best_snr, _ = best
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
