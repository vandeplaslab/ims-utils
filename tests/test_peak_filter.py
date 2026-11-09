import numpy as np
import pytest

pytest.importorskip("pandas")

from ims_utils.peak_filter import (
    PeakFilter,
    filter_groups,
    filter_groups_first,
    get_info,
    get_weights_for_indices,
    get_weights_for_indicies,
)


def test_reset_clears_matrix_and_other_selections() -> None:
    peaks = PeakFilter(np.array([100.1, 200.2, 300.3]), np.array([2.0, 4.0, 6.0]))
    peaks._indices_by_matrix = np.array([0, 2])
    peaks.filter_by_mass_defect(0.15, 0.25)
    np.testing.assert_array_equal(peaks.indices, [0, 2])
    np.testing.assert_allclose(peaks.removed_mzs, [200.2])
    peaks.reset_filters()
    assert peaks._indices_by_matrix is None
    assert peaks._groups is None
    with pytest.raises(ValueError, match="No indices"):
        _ = peaks.indices
    peaks.filter_by_mass_defect(0.25, 0.35)
    np.testing.assert_array_equal(peaks.indices, [0, 1])


def test_merge_respects_maximum_group_size() -> None:
    peaks = PeakFilter(np.array([100.0, 100.001, 100.002, 100.003]), np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(peaks.filter_by_mz_merge(max_in_group=2), [1, 3])
    np.testing.assert_array_equal(peaks.remaining_mzs, [100.001, 100.003])
    assert peaks.is_removed(100.0)


def test_matrix_filter_uses_packaged_masses() -> None:
    import pandas as pd

    from ims_utils.assets import find_matrix

    path = find_matrix("mapca", "positive")
    matrix_mz = pd.read_csv(path).mzs.iloc[0]
    peaks = PeakFilter(np.array([matrix_mz, matrix_mz + 0.1]), np.array([1.0, 2.0]), polarity="positive")
    np.testing.assert_array_equal(peaks.filter_by_matrix("mapca"), [1])
    with pytest.raises(FileNotFoundError):
        peaks.filter_by_matrix("missing")


def test_mass_filter_empty_rules_and_intersection() -> None:
    peaks = PeakFilter(np.array([100.1, 200.2, 300.3]), np.array([1.0, 5.0, 3.0]))
    np.testing.assert_array_equal(peaks.filter_by_kendrick_mass("CH2", []), [0, 1, 2])
    np.testing.assert_array_equal(peaks.filter_by_mass_defect(0.15, 0.25), [0, 2])
    x, y = peaks.get_mzs_and_intensities_for(peaks.indices)
    np.testing.assert_allclose(x, [100.1, 300.3])
    np.testing.assert_array_equal(y, [1.0, 3.0])


def test_group_helpers_choose_highest_and_preserve_legacy_spelling() -> None:
    groups = [([0, 1], [100.0, 101.0]), ([2], [200.0])]
    intensity = np.array([1.0, 5.0, 3.0])
    np.testing.assert_array_equal(filter_groups(groups, intensity), [1, 2])
    np.testing.assert_array_equal(filter_groups_first(groups), [0, 2])
    assert get_info([0, 1], intensity) == (1, ["removed", "selected"])
    weights = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.7], [0.3, 0.7, 1.0]])
    for function in (get_weights_for_indices, get_weights_for_indicies):
        np.testing.assert_allclose(function(weights, [0, 1, 2]), [1.0, 0.8, 0.7])
        assert function(weights, []).size == 0


def test_isotope_groups_map_back_to_original_indices() -> None:
    peaks = PeakFilter(np.array([100.0, 100.5, 101.00235, 102.0047]), np.array([1.0, 9.0, 2.0, 3.0]))
    peaks.filter_by_mass_defect(0.4, 0.6)
    assert peaks.group_into_isotope_groups()[0][0] == [0, 2, 3]
    assert peaks.get_groups(sort_by="count")[0][0] == [0, 2, 3]
