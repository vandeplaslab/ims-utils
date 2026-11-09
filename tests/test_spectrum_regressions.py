from collections.abc import Callable

import numpy as np
import pytest
from scipy.signal import peak_prominences

from ims_utils.spectrum import (
    estimate_peak_prominence,
    find_between_batch,
    get_mzs_for_tol,
    group_peaks_by_tol,
    local_estimate_noise,
    merge_peaks_by_ppm,
    merge_peaks_by_tol,
    ppm_error,
    ppm_to_delta_mass,
    select_nearest,
)


def test_local_noise_mz_selection_matches_nearest_indices() -> None:
    mz = np.arange(100.0, 110.0)
    intensity = np.array([1.0, 5.0, 2.0, 8.0, 1.0, 6.0, 2.0, 7.0, 1.0, 3.0])
    actual = local_estimate_noise(mz, intensity, [99.0, 101.1, 103.1, 105.1, 109.0, 110.0], peaks_kind="mz")
    expected = local_estimate_noise(mz, intensity, [0, 1, 3, 5, 9, 9])
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference)


def test_local_noise_empty_and_invalid_peaks() -> None:
    for output in local_estimate_noise(np.array([]), np.array([]), []):
        assert output.size == 0
    with pytest.raises(ValueError, match="empty"):
        local_estimate_noise(np.array([]), np.array([]), [100.0], peaks_kind="mz")
    with pytest.raises(ValueError, match="out of range"):
        local_estimate_noise(np.array([100.0]), np.array([1.0]), [1])


def test_prominence_matches_reference_for_known_peaks() -> None:
    signal = np.array([0.0, 3.0, 1.0, 5.0, 2.0, 4.0, 0.0])
    indices = np.array([5, 1, 3])
    actual = estimate_peak_prominence(signal, indices)
    expected = peak_prominences(signal, indices)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_mass_windows_and_ppm_conversions() -> None:
    mz = np.array([100.0, 500.0])
    np.testing.assert_allclose(ppm_to_delta_mass(mz, 10), [0.001, 0.005])
    np.testing.assert_allclose(ppm_error(mz * 1.00001, mz), [10, 10])
    lower, upper = get_mzs_for_tol(mz, tol=0.1)
    np.testing.assert_allclose(lower, mz - 0.1)
    np.testing.assert_allclose(upper, mz + 0.1)
    windows = find_between_batch(np.arange(10.0), np.array([1.0, 8.0]), np.array([3.0, 10.0]))
    np.testing.assert_array_equal(windows[0], [1, 2, 3])
    np.testing.assert_array_equal(windows[1], [8, 9])
    np.testing.assert_array_equal(select_nearest(mz, np.array([100.0001, 502.0]), 3), [100.0])


@pytest.mark.parametrize("method", [merge_peaks_by_tol, merge_peaks_by_ppm])
def test_peak_merge_keeps_intensity_and_mobility_paired(method: Callable) -> None:
    mz = np.array([100.0, 100.0001, 200.0])
    x, y, mobility = method(mz, np.array([1.0, 5.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(x, [100.0001, 200.0])
    np.testing.assert_allclose(y, [5.0, 3.0])
    np.testing.assert_allclose(mobility, [0.2, 0.3])
    indices, values = group_peaks_by_tol(mz, 0.001)
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_array_equal(values[1], [200.0])
