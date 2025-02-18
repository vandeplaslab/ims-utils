"""Test filters."""
from pathlib import Path

import numpy as np
import pytest

from ms_utils.filters import FILTER_REGISTER, FilterBase, PpmResampling, register, transform

BASE_PATH = Path(__file__).parent
TEST_PATH = BASE_PATH / "test_data" / "test_mz_peaks_#0.npz"


def get_data():
    """Get data"""
    with np.load(TEST_PATH) as f_ptr:  # type: ignore
        x = f_ptr["x"]
        y = f_ptr["y"]
    return x, y


def test_register():
    @register("TestClass")
    class TestClass(FilterBase):
        def filter(self, x, y):
            return x, y

    assert "TestClass" in FILTER_REGISTER
    TestClass([0], [0])


@pytest.mark.parametrize("name", FILTER_REGISTER.keys())
def test_filter(name):
    x, y = get_data()
    flt = FILTER_REGISTER[name]

    # test call
    _x, _y = flt(x, y)
    assert isinstance(_x, np.ndarray)
    assert isinstance(_y, np.ndarray)
    assert len(_x) == len(_y)

    # test method
    _x, _y = flt.filter(x, y)
    assert isinstance(_x, np.ndarray)
    assert isinstance(_y, np.ndarray)
    assert len(_x) == len(_y)


def test_transform():
    x, y = get_data()
    _x, _y = transform(x, y, ["linear", "gaussian"])
    assert isinstance(_x, np.ndarray)
    assert isinstance(_y, np.ndarray)
    assert len(_x) == len(_y)


def test_ppm():
    x, y = get_data()

    flt = PpmResampling(1, 100, 2000)
    _x, _y = flt(x, y)
    assert isinstance(_x, np.ndarray)
    assert isinstance(_y, np.ndarray)
    assert len(_x) == len(_y)

    with pytest.raises(ValueError):
        PpmResampling(0, 0, 2000)
    with pytest.raises(ValueError):
        PpmResampling(5, 500, 500)
