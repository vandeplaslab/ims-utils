import numpy as np
import pytest

from ims_utils.deisotope import deisotope
from ims_utils.utilities import find_groups_within_bounds, find_groups_within_bounds_weighted, find_highest


@pytest.mark.parametrize("weighted", [False, True])
def test_groups_do_not_reuse_peaks(weighted: bool) -> None:
    mz = np.array([100.0, 100.00001, 101.00235])
    if weighted:
        groups = find_groups_within_bounds_weighted(mz, np.ones((3, 3)), keep_singletons=True)
    else:
        groups = find_groups_within_bounds(mz, keep_singletons=True)
    assert [indices for indices, _ in groups] == [[0, 2], [1]]
    flattened = [index for indices, _ in groups for index in indices]
    assert sorted(flattened) == list(range(3))


@pytest.mark.parametrize("middle", [100.5, 101.00234])
def test_weighted_groups_skip_low_weight_candidate(middle: float) -> None:
    mz = np.array([100.0, middle, 101.00235])
    weights = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    groups = find_groups_within_bounds_weighted(mz, weights)
    assert [indices for indices, _ in groups] == [[0, 2]]


def test_weight_reference_first_or_previous() -> None:
    mz = np.array([100.0, 101.0, 102.0])
    weights = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert find_groups_within_bounds_weighted(mz, weights, distance=1)[0][0] == [0, 1, 2]
    assert find_groups_within_bounds_weighted(mz, weights, distance=1, compare_to_last=False)[0][0] == [0, 1]


def test_groups_ppm_limits_singletons_and_empty() -> None:
    mz = np.array([1000.0, 1001.001, 1002.002, 1008.0])
    groups = find_groups_within_bounds(mz, distance=1, ppm=2, max_in_group=2, keep_singletons=True)
    assert [indices for indices, _ in groups] == [[0, 1], [2], [3]]
    assert find_groups_within_bounds(np.array([])) == []
    assert find_groups_within_bounds_weighted(np.array([]), np.empty((0, 0))) == []
    assert find_highest([0, 1, 2], np.array([1.0, 5.0, 2.0])) == 1
    with pytest.raises(ValueError):
        find_highest([], np.array([]))


def test_two_peak_isotope_group_includes_final_peak() -> None:
    mz, intensity = np.array([100.0, 101.00235]), np.array([10.0, 5.0])
    groups, remaining_mz, remaining_intensity = deisotope(mz, intensity)
    assert len(groups) == 1
    np.testing.assert_allclose(groups[0], [[100.0, 10.0], [101.00235, 5.0]])
    assert remaining_mz == remaining_intensity == []
    np.testing.assert_array_equal(mz, [100.0, 101.00235])


def test_isotope_groups_skip_intervening_peaks_without_losing_duplicates() -> None:
    mz = np.array([100.0, 100.0, 100.5, 101.00235, 102.0047, 110.0])
    intensity = np.arange(1.0, 7.0)
    groups, remaining_mz, remaining_intensity = deisotope(mz, intensity)
    np.testing.assert_allclose(groups[0][:, 0], [100.0, 101.00235, 102.0047])
    np.testing.assert_allclose(remaining_mz, [100.0, 100.5, 110.0])
    np.testing.assert_allclose(remaining_intensity, [2.0, 3.0, 6.0])


@pytest.mark.parametrize("mz", [np.array([]), np.array([100.0])])
def test_deisotope_empty_and_singleton(mz: np.ndarray) -> None:
    groups, remaining, _ = deisotope(mz, np.ones(len(mz)))
    assert groups == []
    np.testing.assert_array_equal(remaining, mz)


def test_deisotope_zero_iterations_and_invalid_lengths() -> None:
    mz, intensity = np.array([100.0, 101.00235]), np.ones(2)
    groups, remaining, _ = deisotope(mz, intensity, n_iter=0)
    assert groups == []
    assert remaining is mz
    with pytest.raises(ValueError, match="same length"):
        deisotope(mz, intensity[:1])
    with pytest.raises(ValueError, match="non-negative"):
        deisotope(mz, intensity, n_iter=-1)
