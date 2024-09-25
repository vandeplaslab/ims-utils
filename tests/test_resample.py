"""Resample functionality."""

import numpy as np

from spec_utils.resample import get_resampler_from_mzs


def test_resampler():
    mz_xs = [np.arange(1, 100), np.arange(1, 120)]
    resampler = get_resampler_from_mzs(mz_xs)
    assert resampler is not None, "resampler is None"
    assert resampler.mz_start == 1.0, "mz_min is not 0.0"
    assert resampler.mz_end == 119.0, "mz_max is not 119.0"
    assert resampler.mz_new is not None, "mz_new is None"

    new_xs = np.arange(1, 51)
    new_ys = np.random.randint(0, 100, 50)
    resampled_xs, resampled_ys = resampler(new_xs, new_ys)
    assert resampled_xs.size == resampled_ys.size, "resampled_xs and resampled_ys have different sizes"
    assert resampled_xs.size != new_xs.size, "resampled_xs and new_xs have the same size"
