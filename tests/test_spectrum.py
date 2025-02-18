"""Test imimspy.processing.spectrum.py functions"""
from pathlib import Path

import numpy as np
import pytest

from ms_utils.smooth import gaussian
from ms_utils.spectrum import maxima_centroid, parabolic_centroid

BASE_PATH = Path(__file__).parent
TEST_PATH = BASE_PATH / "test_data" / "test_mz_peaks_#0.npz"


def get_data():
    """Get data"""
    with np.load(TEST_PATH) as f_ptr:
        x = f_ptr["x"]
        y = f_ptr["y"]
    return x, y


@pytest.mark.parametrize("threshold", (50, 500, 1000))
def test_parabolic_centroid(threshold):
    x, y = get_data()
    cx, cy = parabolic_centroid(x, y, threshold)
    assert len(cx) == len(cy)
    assert len(cx) < len(x)
    assert cx[0] >= x[0]
    assert cx[-1] <= x[-1]


def test_maxima_centroid():
    x, y = get_data()
    cx, cy = maxima_centroid(x, y)
    assert len(cx) == len(cy)
    assert len(cx) < len(x)
    assert cx[0] >= x[0]
    assert cx[-1] <= x[-1]


@pytest.mark.parametrize("sigma", (1, 5))
def test_smooth_spectrum(sigma):
    x, y = get_data()
    sy = gaussian(y, sigma)
    assert len(sy) == len(x)
