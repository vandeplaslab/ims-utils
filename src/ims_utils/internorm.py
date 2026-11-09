"""Inter normalization utilities."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


class CentroidMeanFoldInterNorm:
    """Collect centroid datasets and calculate multiplicative inter-normalization scales.

    Each dataset is a samples-by-features array. Feature counts must agree across
    datasets. :meth:`finalize` returns one scale per input dataset in collection
    order; multiply each dataset by its scale to apply the normalization.
    """

    def __init__(self) -> None:
        self.tmp_centroids: list[np.ndarray] = []
        self.tmo_names: list[str] = []
        self.centroids_ref = None
        self.scales = None

    def __call__(self, name: str, centroid: np.ndarray) -> None:
        """Collect a named samples-by-features dataset for normalization.

        Parameters
        ----------
        name : str
            Dataset name retained in the finalized output.
        centroid : np.ndarray
            Non-empty two-dimensional array. Its feature count must match all
            previously collected datasets.
        """
        centroid_ref = _get_mean_fold_change(centroid)
        if self.tmp_centroids and centroid_ref.shape != self.tmp_centroids[0].shape:
            raise ValueError("All centroids must have the same number of features.")
        self.tmp_centroids.append(centroid_ref)
        self.tmo_names.append(name)

    def finalize(self, as_dict: bool = False) -> tuple[list[str], np.ndarray] | dict[str, float]:
        """Calculate and return multiplicative scales in collection order.

        Parameters
        ----------
        as_dict : bool
            If true, return a mapping from each collected name to its scale.

        Returns
        -------
        tuple[list[str], np.ndarray] | dict[str, float]
            Dataset names and scales, or a name-to-scale mapping. Scales are
            normalized so their minimum is 1. At least two datasets are required.
        """
        if not self.tmp_centroids:
            raise ValueError("No centroids found.")
        if len(self.tmp_centroids) == 1:
            raise ValueError("Only one centroid found. Inter-normalization is not possible.")

        centroids_array = np.concatenate(self.tmp_centroids, axis=0)
        scales, _ = _get_internorm_scales(centroids_array)
        scales = 1 / scales
        scales /= np.min(scales)
        if as_dict:
            return scales_to_dict(self.tmo_names, scales)
        return self.tmo_names, scales


def calculate_mfc_inter_normalization(centroids: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Calculate inter-normalization vector of values.

    Parameters
    ----------
    centroids : dict
        Dictionary of centroids (2D array of shape [n_px, n_features])

    Returns
    -------
    tuple
        List of keys and inter-normalization vector.
        There will be a single scalar value for each centroid which can be used to multiply the centroid data.
    """
    if not centroids:
        raise ValueError("No centroids found.")
    if len(centroids) == 1:
        raise ValueError("Only one centroid found. Inter-normalization is not possible.")

    obj = CentroidMeanFoldInterNorm()
    for name, centroid in tqdm(centroids.items(), desc="Collecting centroids for inter-normalization..."):
        obj(name, centroid)
    return obj.finalize()


def _get_mean_fold_change(centroid: np.ndarray) -> np.ndarray:
    """Get scaling factors based on median values."""
    centroid = np.asarray(centroid)
    if centroid.ndim != 2 or 0 in centroid.shape:
        raise ValueError("Centroids must be non-empty 2D arrays of shape (pixels, features).")
    # Compute the median while handling NaNs
    centroid_ref = np.nanmedian(centroid, axis=0) if np.isnan(centroid).any() else np.median(centroid, axis=0)
    centroid_ref = centroid_ref.astype(np.float32)
    # Replace zeros with NaNs
    centroid_ref[centroid_ref == 0] = np.nan
    return centroid_ref.reshape(1, -1)  # Ensure it remains 2D


def _mean_fold_change(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate median fold change value."""
    centroids_ref = np.nanmedian(x, axis=0)
    centroids_ref[centroids_ref == 0] = np.nan
    scaling_factors = np.nanmedian(x / centroids_ref, axis=1)
    return scaling_factors.astype(np.float32), centroids_ref.astype(np.float32)


def _get_internorm_scales(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate inter-sample scales without collapsing singleton feature dimensions."""
    scaling_factors, centroids_ref = _mean_fold_change(x)

    # Ensure stability in calculations
    scaling_factors = np.nan_to_num(scaling_factors, nan=1.0, posinf=1.0, neginf=1.0)
    scaling_factors[scaling_factors == 0] = 1
    # Normalize by median
    median_scale = np.nanmedian(scaling_factors)
    scaling_factors /= median_scale if median_scale else 1
    return scaling_factors.astype(np.float32), centroids_ref.astype(np.float32)


def calculate_mean_inter_normalization(centroids: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return names and mean-based multiplicative scales in dictionary order.

    Values have shape (pixels, features). Each scale multiplies its dataset;
    scales are normalized so the smallest is one. Inputs are not modified.
    """
    if not centroids:
        raise ValueError("No centroids found.")
    if len(centroids) == 1:
        raise ValueError("Only one centroid found. Inter-normalization is not possible.")

    # Compute the mean vector for each centroid
    centroids_ref = np.array([np.mean(centroid, axis=0) for centroid in centroids.values()])
    # Compute scaling factors
    overall_mean = np.mean(centroids_ref)
    scales = overall_mean / np.mean(centroids_ref, axis=1)
    # Handle NaNs and zeros
    scales = np.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
    scales[scales == 0] = 1
    # Normalize so that the minimum scale is 1
    scales /= np.min(scales)
    return list(centroids.keys()), scales


def calculate_median_inter_normalization(centroids: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return multiplicative scales from medians of per-feature pixel means.

    Values have shape (pixels, features). Names follow dictionary order and
    the smallest scale is one. This method retains its mean-over-pixels step;
    it does not take a median of all raw pixels. Inputs are not modified.
    """
    if not centroids:
        raise ValueError("No centroids found.")
    if len(centroids) == 1:
        raise ValueError("Only one centroid found. Inter-normalization is not possible.")

    # Compute the mean vector for each centroid
    centroids_ref = np.array([np.mean(centroid, axis=0) for centroid in centroids.values()])
    # Compute scaling factors
    overall_median = np.median(centroids_ref)
    scales = overall_median / np.median(centroids_ref, axis=1)
    # Handle NaNs and zeros
    scales = np.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
    scales[scales == 0] = 1
    # Normalize so that the minimum scale is 1
    scales /= np.min(scales)
    return list(centroids.keys()), scales


def calculate_tic_inter_normalization(tics: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return names and scales based on mean TIC per dataset, in dictionary order.

    Each value is a one-dimensional array of total ion counts per pixel.
    Scales multiply the original data and have a minimum of one.
    """
    if not tics:
        raise ValueError("No TICs found.")
    if len(tics) == 1:
        raise ValueError("Only one TIC found. Inter-normalization is not possible.")
    # Compute mean of each TIC efficiently using a list comprehension
    tic_ref = np.array([np.mean(tic) for tic in tics.values()], dtype=np.float32)
    # Compute the median and normalization factors
    median = np.median(tic_ref)
    scales = median / tic_ref
    # Normalize scales, ensuring no NaNs or zeros
    scales = np.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
    scales[scales == 0] = 1
    # Ensure minimum scale is 1
    scales /= np.min(scales)
    return list(tics.keys()), scales


def scales_to_dict(names: list[str], scales: np.ndarray) -> dict[str, float]:
    """Convert scales to dictionary."""
    return {name: float(scale) for name, scale in zip(names, scales)}


def log_scales(scales: dict[str, float], method: str) -> None:
    """Log scales."""
    from loguru import logger

    logger.info(f"Inter-normalization scales using method '{method}':")
    for name, scale in scales.items():
        logger.trace(f"  {name}: {scale:.4f}")
