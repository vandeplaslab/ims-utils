"""Inter normalization utilities."""
from __future__ import annotations
import numpy as np
from tqdm import tqdm


class CentroidMeanFoldInterNorm:
    """Centroid inter-normalization methods."""

    def __init__(self):
         self.tmp_centroids = []
         self.tmo_names = []
         self.centroids_ref = None
         self.scales = None

    def __call__(self, name: str, centroid: np.ndarray) -> None:
        """Collect centroid data."""
        self.tmp_centroids.append(_get_mean_fold_change(centroid))
        self.tmo_names.append(name)

    def finalize(self, as_dict: bool = False) -> tuple[list[str], np.ndarray] | dict[str, float]:
        """Finalize inter-normalization calculation."""
        if not self.tmp_centroids:
            raise ValueError("No centroids found.")
        if len(self.tmp_centroids) == 1:
            raise ValueError("Only one centroid found. Inter-normalization is not possible.")

        centroids_array = np.array(self.tmp_centroids, dtype=np.float32).squeeze()
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
    """Intra-sample normalization."""
    scaling_factors, centroids_ref = _mean_fold_change(x)

    # Ensure stability in calculations
    scaling_factors = np.nan_to_num(scaling_factors, nan=1.0, posinf=1.0, neginf=1.0)
    scaling_factors[scaling_factors == 0] = 1
    # Normalize by median
    median_scale = np.nanmedian(scaling_factors)
    scaling_factors /= median_scale if median_scale else 1
    return scaling_factors.astype(np.float32).squeeze(), centroids_ref.astype(np.float32).squeeze()


def calculate_mean_inter_normalization(centroids: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Calculate inter-normalization vector of values."""
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
    """Calculate inter-normalization vector of values."""
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
    """Calculate inter-normalization vector of values."""
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