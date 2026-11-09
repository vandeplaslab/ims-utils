"""Test inter normalization"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ims_utils.internorm import (
    CentroidMeanFoldInterNorm,
    calculate_mean_inter_normalization,
    calculate_median_inter_normalization,
    calculate_mfc_inter_normalization,
    calculate_tic_inter_normalization,
    scales_to_dict,
)

ARRAY_2D = np.random.default_rng(0).integers(1, 255, (100, 10)) * 1.0
DATA_2D = {"a": ARRAY_2D, "b": ARRAY_2D * 2.0, "c": ARRAY_2D * 0.5}

ARRAY_1D = np.random.default_rng(1).integers(1, 255, (100,)) * 1.0
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


def test_calculate_median_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_median_inter_normalization(DATA_2D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0


def test_calculate_tic_inter_normalization():
    """Test calculate_mfc_inter_normalization"""
    names, scales = calculate_tic_inter_normalization(DATA_1D)
    assert len(names) == 3
    assert len(scales) == 3
    assert scales.min() == 1.0
    assert scales.max() == 4.0


def test_mfc_single_feature_preserves_order() -> None:
    names, scales = calculate_mfc_inter_normalization(
        {"second": np.array([[2.0], [4.0]]), "first": np.array([[1.0], [2.0]])}
    )
    assert names == ["second", "first"]
    np.testing.assert_allclose(scales, [1.0, 2.0])


def test_mfc_rejected_input_does_not_poison_collector() -> None:
    collector = CentroidMeanFoldInterNorm()
    collector("a", np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="same number"):
        collector("bad", np.ones((2, 3)))
    collector("b", np.array([[2.0, 4.0], [6.0, 8.0]]))
    assert collector.finalize(as_dict=True) == pytest.approx({"a": 2.0, "b": 1.0})
    assert scales_to_dict(["b", "a"], np.array([1.0, 2.0])) == {"b": 1.0, "a": 2.0}


@pytest.mark.parametrize("data", [np.ones(2), np.empty((0, 2)), np.empty((2, 0))])
def test_mfc_rejects_invalid_centroid_shapes(data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty 2D"):
        CentroidMeanFoldInterNorm()("bad", data)


@pytest.mark.parametrize(
    "method",
    [
        calculate_mfc_inter_normalization,
        calculate_mean_inter_normalization,
        calculate_median_inter_normalization,
        calculate_tic_inter_normalization,
    ],
)
def test_internorm_requires_multiple_datasets(method: Callable[[dict[str, np.ndarray]], object]) -> None:
    with pytest.raises(ValueError, match="No"):
        method({})
    with pytest.raises(ValueError, match="Only one"):
        method({"only": np.ones((2, 2))})


def test_collector_requires_multiple_datasets() -> None:
    collector = CentroidMeanFoldInterNorm()
    with pytest.raises(ValueError, match="No centroids"):
        collector.finalize()
    collector("only", np.ones((2, 1)))
    with pytest.raises(ValueError, match="Only one"):
        collector.finalize()
