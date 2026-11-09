"""Resample functionality."""

from __future__ import annotations

import typing as ty

import numpy as np

from ims_utils.filters import MzResampling, PpmResampling
from ims_utils.spectrum import ppm_diff


class Object(ty.Protocol):
    """Reader providing a one-dimensional, ascending m/z axis."""

    mz_x: np.ndarray
    mz_range: tuple[float, float]


def _get_spacing(mz_xs: ty.Iterable[np.ndarray]) -> float:
    """Get spacing from spectra."""
    min_values = []
    for mz_x in mz_xs:
        differences = ppm_diff(mz_x)
        positive = differences[np.isfinite(differences) & (differences > 0)]
        if positive.size:
            min_values.append(float(positive.min()))
    if not min_values:
        raise ValueError("At least one m/z axis must contain two distinct, increasing values.")
    return min(min_values)


def get_spacing(readers: ty.Iterable[Object]) -> float:
    """Return the smallest positive adjacent spacing in ppm across reader axes."""
    mz_xs = [reader.mz_x for reader in readers]
    return _get_spacing(mz_xs)


def _get_mass_range(mz_xs: ty.Iterable[np.ndarray]) -> tuple[float, float]:
    """Return the union of the supplied m/z ranges."""
    mz_ranges = [(np.min(mz_x), np.max(mz_x)) for mz_x in mz_xs if len(mz_x)]
    if not mz_ranges:
        raise ValueError("At least one non-empty m/z axis is required.")
    return float(np.min(mz_ranges)), float(np.max(mz_ranges))


def get_mass_range(readers: ty.Iterable[Object]) -> tuple[float, float]:
    """Return the minimum and maximum m/z across non-empty reader axes."""
    mz_xs = [reader.mz_x for reader in readers]
    return _get_mass_range(mz_xs)


def get_resampler_from_mzs(mz_xs: ty.Iterable[np.ndarray]) -> PpmResampling:
    """Build a ppm resampler spanning ascending input axes, including generators.

    The smallest positive adjacent ppm spacing is used. Empty or singleton axes
    cannot establish spacing; at least one axis must have two distinct values.
    """
    mz_xs = list(mz_xs)
    ppm_spacing = _get_spacing(mz_xs)
    mz_min, mz_max = _get_mass_range(mz_xs)
    return PpmResampling(ppm_spacing, mz_min, mz_max)


def get_resampler_from_mz(mz_x: np.ndarray, original_scale: bool = False, original_dtype: bool = False) -> MzResampling:
    """Build a resampler targeting ``mz_x``; optionally preserve intensity maximum and dtype."""
    return MzResampling(mz_x, original_scale, original_dtype)


def get_resampler(readers: ty.Iterable[Object]) -> PpmResampling:
    """Build a common ppm resampler from readers exposing ascending ``mz_x`` arrays."""
    mz_xs = [reader.mz_x for reader in readers]
    return get_resampler_from_mzs(mz_xs)


def resample_spectra(readers: ty.Iterable[Object], spectra: list[np.ndarray]) -> list[np.ndarray]:
    """Replace spectra in place with intensity arrays on a common ppm grid.

    Readers and spectra must have matching lengths. Each reader provides the
    ascending m/z axis for the corresponding one-dimensional intensity array.
    The original list is returned; use ``get_resampler`` to obtain the grid.
    """
    readers = list(readers)
    if len(readers) != len(spectra):
        raise ValueError("Readers and spectra must have the same length.")
    if not spectra:
        return spectra
    resampler = get_resampler(readers)
    for i, y in enumerate(spectra):
        _, spectra[i] = resampler(readers[i].mz_x, y)
    return spectra
