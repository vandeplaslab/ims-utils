from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ims_utils.filters import IndexRecalibrate, MeanBelowMeanFilter, MzResampling, retype
from ims_utils.resample import _get_mass_range, _get_spacing, get_resampler_from_mzs, resample_spectra


def test_resampler_accepts_generator() -> None:
    axes = [np.array([100.0, 101.0, 102.0]), np.array([101.0, 102.0, 103.0])]
    actual = get_resampler_from_mzs(axis for axis in axes)
    expected = get_resampler_from_mzs(axes)
    np.testing.assert_array_equal(actual.mz_new, expected.mz_new)
    assert (actual.mz_start, actual.mz_end) == (100.0, 103.0)


def test_resample_spectra_returns_intensities_and_updates_list() -> None:
    axis = np.array([100.0, 101.0, 102.0])
    readers = [SimpleNamespace(mz_x=axis), SimpleNamespace(mz_x=axis)]
    spectra = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])]
    target = get_resampler_from_mzs([axis, axis]).mz_new
    actual = resample_spectra(iter(readers), spectra)
    assert actual is spectra
    for factor, intensity in enumerate(actual, 1):
        assert isinstance(intensity, np.ndarray)
        np.testing.assert_allclose(intensity, np.interp(target, axis, factor * np.arange(1.0, 4.0)))


def test_resample_spectra_validates_lengths_before_mutating() -> None:
    spectra = [np.ones(3)]
    with pytest.raises(ValueError, match="same length"):
        resample_spectra([], spectra)
    np.testing.assert_array_equal(spectra[0], np.ones(3))
    assert resample_spectra([], []) == []


@pytest.mark.parametrize("axes", [[], [np.array([])], [np.array([100.0])], [np.array([100.0, 100.0])]])
def test_spacing_requires_a_positive_step(axes: list[np.ndarray]) -> None:
    with pytest.raises(ValueError, match="distinct"):
        _get_spacing(axes)


def test_spacing_ignores_duplicate_steps() -> None:
    assert _get_spacing([np.array([100.0, 100.0, 101.0])]) == pytest.approx(10_000.0)
    assert _get_mass_range([np.array([]), np.array([100.0, 102.0])]) == (100.0, 102.0)
    with pytest.raises(ValueError, match="non-empty"):
        _get_mass_range([])


@pytest.mark.parametrize("target", [np.arange(99.0, 106.0), np.array([101.0, 103.0]), np.array([104.0, 105.0])])
def test_mz_resampling_preserves_source_endpoints(target: np.ndarray) -> None:
    axis = np.arange(100.0, 105.0)
    intensity = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    _, actual = MzResampling(target)(axis, intensity)
    np.testing.assert_allclose(actual, np.interp(target, axis, intensity))


def test_resample_zero_signal_and_preserve_dtype() -> None:
    axis = np.arange(100.0, 105.0)
    original = np.zeros(5, dtype=np.uint16)
    with np.errstate(all="raise"):
        _, actual = MzResampling(axis, original_scale=True, original_dtype=True)(axis, original)
    np.testing.assert_array_equal(actual, original)
    assert actual.dtype == original.dtype
    np.testing.assert_array_equal(retype(np.array([1.2, 1.8]), np.dtype("uint16")), [1, 2])


def test_resample_original_scale() -> None:
    _, result = MzResampling(np.array([100.5, 101.5]), original_scale=True)(
        np.array([100.0, 101.0, 102.0]), np.array([0.0, 10.0, 0.0])
    )
    np.testing.assert_allclose(result, [10.0, 10.0])


def test_mean_below_mean_uses_signal_values() -> None:
    intensity = np.array([1.0, 2.0, 10.0, 20.0])
    _, actual = MeanBelowMeanFilter()(np.arange(4.0), intensity)
    np.testing.assert_array_equal(actual, [0.0, 2.0, 10.0, 20.0])
    _, constant = MeanBelowMeanFilter()(np.arange(4.0), np.ones(4))
    np.testing.assert_array_equal(constant, np.ones(4))


def test_zero_index_recalibration_is_identity() -> None:
    axis, intensity = np.arange(100.0, 105.0), np.arange(5.0)
    shifted, output = IndexRecalibrate(0)(axis, intensity)
    np.testing.assert_array_equal(shifted, axis)
    np.testing.assert_array_equal(axis, np.arange(100.0, 105.0))
    assert output is intensity


@pytest.mark.parametrize("shift", [-10, -2, -1, 1, 2, 10])
def test_index_recalibration_extrapolates_without_duplicate_coordinates(shift: int) -> None:
    axis = np.arange(100.0, 110.0)
    result, _ = IndexRecalibrate(shift)(axis, np.ones(10))
    np.testing.assert_allclose(result, axis - shift)


def test_index_recalibration_nonuniform_axis() -> None:
    axis = np.array([100.0, 102.0, 105.0, 109.0])
    left, _ = IndexRecalibrate(2)(axis, np.ones(4))
    right, _ = IndexRecalibrate(-2)(axis, np.ones(4))
    np.testing.assert_allclose(left, [96.0, 98.0, 100.0, 102.0])
    np.testing.assert_allclose(right, [105.0, 109.0, 113.0, 117.0])
    with pytest.raises(ValueError, match="two m/z"):
        IndexRecalibrate(1)(np.array([100.0]), np.ones(1))
