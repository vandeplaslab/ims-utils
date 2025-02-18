"""Test inter normalization"""

import numpy as np

from ims_utils.internorm import (
    calculate_mean_inter_normalization,
    calculate_median_inter_normalization,
    calculate_mfc_inter_normalization,
    calculate_tic_inter_normalization,
)

ARRAY_2D = np.random.randint(0, 255, (100, 10)) * 1.0
DATA_2D = {"a": ARRAY_2D, "b": ARRAY_2D * 2.0, "c": ARRAY_2D * 0.5}

ARRAY_1D = np.random.randint(0, 255, (100,)) * 1.0
DATA_1D = {"a": ARRAY_1D, "b": ARRAY_1D * 2.0, "c": ARRAY_1D * 0.5}


def test_calculate_mfc_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_mfc_inter_normalization(DATA_2D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0


def test_calculate_mean_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_mean_inter_normalization(DATA_2D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0


def calculate_median_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_median_inter_normalization(DATA_2D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0


def calculate_tic_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_tic_inter_normalization(DATA_1D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0
