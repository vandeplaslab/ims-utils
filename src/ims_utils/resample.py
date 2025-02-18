"""Resample functionality."""

from __future__ import annotations

import typing as ty

import numpy as np
from koyo.spectrum import ppm_diff

from ims_utils.filters import PpmResampling


class Object(ty.Protocol):
    """Objects."""

    mz_x: np.ndarray
    mz_range: tuple[float, float]


def _get_spacing(mz_xs: ty.Iterable[np.ndarray]) -> float:
    """Get spacing from spectra."""
    min_values = np.array([np.min(ppm_diff(mz_x)) for mz_x in mz_xs])
    ppm_spacing = np.min(min_values[min_values > 0]) or min_values.mean() or 1.0
    return ppm_spacing


def get_spacing(readers: ty.Iterable[Object]) -> float:
    """Get minimum spacing between spectra."""
    mz_xs = [reader.mz_x for reader in readers]
    return _get_spacing(mz_xs)


def _get_mass_range(mz_xs: ty.Iterable[np.ndarray]) -> tuple[float, float]:
    """Get minimum spacing between spectra."""
    mz_ranges = [(np.min(mz_x), np.max(mz_x)) for mz_x in mz_xs]
    return np.min(mz_ranges), np.max(mz_ranges)


def get_mass_range(readers: ty.Iterable[Object]) -> tuple[float, float]:
    """Get minimum spacing between spectra."""
    mz_xs = [reader.mz_x for reader in readers]
    return _get_mass_range(mz_xs)


def get_resampler_from_mzs(mz_xs: ty.Iterable[np.ndarray]) -> PpmResampling:
    """Get resampler."""
    # we need to convert to list so can be indexed, and we need to be able to iterate twice
    ppm_spacing = _get_spacing(mz_xs)
    mz_min, mz_max = _get_mass_range(mz_xs)
    return PpmResampling(ppm_spacing, mz_min, mz_max)


def get_resampler(readers: ty.Iterable[Object]) -> PpmResampling:
    """Get resampler."""
    # we need to convert to list so can be indexed, and we need to be able to iterate twice
    mz_xs = [reader.mz_x for reader in readers]
    return get_resampler_from_mzs(mz_xs)


def resample_spectra(readers: ty.Iterable[Object], spectra: list[np.ndarray]) -> list[np.ndarray]:
    """Resample spectra."""
    # we need to convert to list so can be indexed
    readers = list(readers)  # type: ignore
    resampler = get_resampler(readers)
    for i, y in enumerate(spectra):
        spectra[i] = resampler(readers[i].mz_x, y)
    return spectra
