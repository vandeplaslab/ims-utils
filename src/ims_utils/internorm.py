"""Inter normalization utilities."""

import numpy as np
from tqdm import tqdm


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

    # Compute mean fold changes in a vectorized manner
    centroids_ref = np.array(
        [
            _get_mean_fold_change(centroid)
            for centroid in tqdm(centroids.values(), desc="Calculating inter-normalization vector...")
        ],
        dtype=np.float32,
    ).squeeze()
    # Compute scaling factors
    scales, _ = _get_internorm_scales(centroids_ref)
    # Normalize to the minimum value
    scales = 1 / scales
    scales /= np.min(scales)
    return list(centroids.keys()), scales


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


def calculate_mean_inter_normalization(centroids: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Calculate inter-normalization vector of values."""
    if not centroids:
        raise ValueError("No centroids found.")

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
