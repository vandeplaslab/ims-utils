import numpy as np
import pytest
from scipy.integrate import trapezoid

from ims_utils.profile import centroid_to_profile


@pytest.mark.parametrize("mode", ["fwhm", "resolving_power"])
def test_sort_keeps_widths_paired_with_centroids(mode: str) -> None:
    mz = np.array([200.0, 100.0])
    height = np.array([3.0, 5.0])
    widths = np.array([2.0, 1.0]) if mode == "fwhm" else np.array([2000.0, 5000.0])
    grid = np.linspace(95.0, 205.0, 22001)
    _, actual, sigma = centroid_to_profile(mz, height, sort=True, mz_grid=grid, **{mode: widths})
    _, expected, expected_sigma = centroid_to_profile(mz[::-1], height[::-1], mz_grid=grid, **{mode: widths[::-1]})
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(sigma, expected_sigma)
    np.testing.assert_allclose(actual[[1000, 21000]], [5.0, 3.0], atol=1e-12)


def test_profile_area_and_height_have_known_gaussian_normalization() -> None:
    grid = np.linspace(99.0, 101.0, 20001)
    _, area, sigma = centroid_to_profile(
        np.array([100.0]), np.array([5.0]), fwhm=0.1, mz_grid=grid, intensity_mode="area"
    )
    assert trapezoid(area, grid) == pytest.approx(5.0, rel=1e-6)
    assert area[10000] == pytest.approx(5.0 / (sigma[0] * np.sqrt(2 * np.pi)))
    _, height, _ = centroid_to_profile(np.array([100.0]), np.array([5.0]), fwhm_func=lambda _: 0.1, mz_grid=grid)
    assert height[10000] == pytest.approx(5.0)


def test_profile_empty_input_with_grid() -> None:
    grid = np.arange(100.0, 105.0)
    output_grid, intensity, sigma = centroid_to_profile(np.array([]), np.array([]), mz_grid=grid)
    np.testing.assert_array_equal(output_grid, grid)
    np.testing.assert_array_equal(intensity, np.zeros(5))
    assert sigma.size == 0
    with pytest.raises(ValueError, match="Empty"):
        centroid_to_profile(np.array([]), np.array([]))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"fwhm": 1, "resolving_power": 1000},
        {"fwhm": -1},
        {"fwhm": np.nan},
        {"fwhm": np.array([1.0, 2.0])},
        {"fwhm": 1, "bin_width": 0},
        {"fwhm": 1, "window": 0},
        {"fwhm": 1, "points_per_fwhm": 0},
        {"fwhm": 1, "mz_grid": np.array([101.0, 100.0])},
        {"fwhm": 1, "intensity_mode": "unknown"},
    ],
)
def test_profile_rejects_invalid_width_or_grid_controls(options: dict) -> None:
    with pytest.raises(ValueError):
        centroid_to_profile(np.array([100.0]), np.array([1.0]), **options)
