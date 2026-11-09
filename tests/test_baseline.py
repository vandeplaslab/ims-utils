import numpy as np
import pytest
from scipy.ndimage import median_filter, white_tophat

from ims_utils.baseline import als, arpls, baseline_1d, curve, linear, median, polynomial, tophat


def test_polynomial_fits_known_baseline_without_modifying_input() -> None:
    axis = np.linspace(0.0, 1.0, 101)
    signal = 2.0 + 3.0 * axis + axis**2
    original = signal.copy()
    np.testing.assert_allclose(polynomial(signal, deg=2), original, atol=1e-10)
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_allclose(baseline_1d(signal, "polynomial", poly_order=2), 0.0, atol=1e-10)


def test_curve_preserves_fractional_baseline() -> None:
    signal = np.full(20, 0.5)
    np.testing.assert_allclose(curve(signal, 2), 0.0, atol=1e-14)
    np.testing.assert_allclose(baseline_1d(signal, "curved", curved_window=2), 0.0, atol=1e-14)
    with pytest.raises(ValueError):
        curve(signal, 0)


def test_linear_threshold_and_input_copy() -> None:
    signal = np.array([0.5, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(baseline_1d(signal, threshold=1.0), [0.0, 0.0, 2.0, 3.0])
    np.testing.assert_array_equal(signal, [0.5, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        linear(signal, -1)
    with pytest.raises(TypeError):
        linear(signal, None)


def test_median_and_tophat_match_scipy() -> None:
    signal = np.array([1.0, 1.0, 8.0, 1.0, 1.0])
    np.testing.assert_array_equal(median(signal, 3), median_filter(signal, 3))
    np.testing.assert_array_equal(tophat(signal, 3), white_tophat(signal, 3))
    np.testing.assert_array_equal(baseline_1d(signal, "median", median_window=3), np.ones(5))
    np.testing.assert_array_equal(baseline_1d(signal, "tophat", tophat_window=3), [0, 0, 7, 0, 0])
    with pytest.raises(ValueError):
        median(signal, 2)


def test_als_and_arpls_recover_flat_background_with_peak() -> None:
    signal = np.full(100, 2.0)
    signal[45:55] = 10.0
    for fitted in [als(signal, 1e5, 0.001), arpls(signal, lam=1e5)]:
        assert np.isfinite(fitted).all()
        np.testing.assert_allclose(fitted, 2.0, atol=0.05)


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_baseline_fits_constant_signals_without_division_warnings(value: float) -> None:
    signal = np.full(20, value)
    for fitted in [polynomial(signal), als(signal, 100.0, 0.01), arpls(signal)]:
        np.testing.assert_allclose(fitted, signal, atol=1e-10)
