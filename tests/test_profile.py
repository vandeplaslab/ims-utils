"""Tests for the centroid to profile conversion."""

import numpy as np

from ims_utils.profile import centroid_to_profile
from ims_utils.spectrum import get_ppm_axis


def test_centroid_to_profile() -> None:
    """Test centroid to profile conversion."""
    # Create a simple centroid spectrum
    mzs = np.array([100, 200, 300], dtype=np.float32)
    intensities = np.array([10, 20, 30], dtype=np.float32)

    # Convert to profile
    profile_mzs, profile_intensities, sigma = centroid_to_profile(
        mzs, intensities, resolving_power=10000, mz_min=50, mz_max=400
    )

    # Check that the output is as expected
    assert profile_mzs.ndim == 1
    assert profile_intensities.ndim == 1
    assert len(profile_mzs) == len(profile_intensities)
    assert len(profile_mzs) > len(mzs)
    assert profile_mzs[0] >= 50
    assert profile_mzs[-1] <= 400

    # Check that the intensities are non-negative
    assert np.all(profile_intensities >= 0)

    # Check that the intensities have peaks at the original m/z values
    for mz in mzs:
        idx = np.argmin(np.abs(profile_mzs - mz))
        assert profile_intensities[idx] > 0


def test_centroid_to_profile_with_ppm() -> None:
    """Test centroid to profile conversion with ppm axis."""
    # Create a simple centroid spectrum
    mzs = np.array([100, 200, 300], dtype=np.float32)
    intensities = np.array([10, 20, 30], dtype=np.float32)

    # Create a ppm axis
    ppm_axis = get_ppm_axis(50, 350, 0.1)

    # Convert to profile
    profile_mzs, profile_intensities, sigma = centroid_to_profile(
        mzs, intensities, resolving_power=10000, mz_grid=ppm_axis
    )

    # Check that the output is as expected
    assert profile_mzs.ndim == 1
    assert profile_intensities.ndim == 1
    assert len(profile_mzs) == len(profile_intensities)
    assert len(profile_mzs) == len(ppm_axis)

    # Check that the intensities are non-negative
    assert np.all(profile_intensities >= 0)

    # Check that the intensities have peaks at the original m/z values
    for mz in mzs:
        idx = np.argmin(np.abs(profile_mzs - mz))
        assert profile_intensities[idx] > 0
