"""Test imimspy.processing.spectrum.py functions"""

from pathlib import Path

import numpy as np
import pytest

from ims_utils.smooth import gaussian
from ims_utils.spectrum import get_ppm_axis, maxima_centroid, parabolic_centroid, ppm_diff, ppm_error

BASE_PATH = Path(__file__).parent
TEST_PATH = BASE_PATH / "_test_data" / "test_mz_peaks_#0.npz"


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


def test_ppm_error():
    ppm = ppm_error(100, 100)
    assert ppm == 0


def test_ppm_diff():
    diff = ppm_diff([100, 100.0001])
    assert diff.size == 1
    np.testing.assert_almost_equal(diff[0], 1, 1)

    diff = ppm_diff([1000, 1000.01])
    assert diff.size == 1
    np.testing.assert_almost_equal(diff[0], 10, 1)


def test_ppm_axis():
    # get new axis
    mz = get_ppm_axis(100, 200, 1)
    # get ppm spacing between each bin
    diff = ppm_diff(mz)
    # make sure that ppm spacing is within 1 dp
    np.testing.assert_almost_equal(diff, 1, 1)

    mz = get_ppm_axis(100, 200, 5)
    # get ppm spacing between each bin
    diff = ppm_diff(mz)
    # make sure that ppm spacing is within 1 dp
    np.testing.assert_almost_equal(diff, 5, 1)
