"""Resample functionality."""

from __future__ import annotations

import typing as ty

import numpy as np
from koyo.spectrum import ppm_diff

from spec_utils.filters import PpmResampling


class Object(ty.Protocol):
    """Objects."""

    mz_x: np.ndarray
    mz_range: tuple[float, float]


def get_spacing(readers: ty.Iterable[Object]) -> float:
    """Get minimum spacing between spectra."""
    min_values = np.array([np.min(ppm_diff(reader.mz_x)) for reader in readers])
    ppm_spacing = np.min(min_values[min_values > 0]) or min_values.mean() or 1.0
    return ppm_spacing


def get_mass_range(readers: ty.Iterable[Object]) -> tuple[float, float]:
    """Get minimum spacing between spectra."""
    mz_ranges = [reader.mz_range for reader in readers]
    return np.min(mz_ranges), np.max(mz_ranges)


def get_resampler(readers: ty.Iterable[Object]) -> PpmResampling:
    """Get resampler."""
    # we need to convert to list so can be indexed, and we need to be able to iterate twice
    readers = list(readers)  # type: ignore
    ppm_spacing = get_spacing(readers)
    mz_min, mz_max = get_mass_range(readers)
    return PpmResampling(ppm_spacing, mz_min, mz_max)


def resample_spectra(readers: ty.Iterable[Object], spectra: list[np.ndarray]) -> list[np.ndarray]:
    """Resample spectra."""
    # we need to convert to list so can be indexed
    readers = list(readers)  # type: ignore
    resampler = get_resampler(readers)
    for i, y in enumerate(spectra):
        spectra[i] = resampler(readers[i].mz_x, y)
    return spectra
