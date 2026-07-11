from __future__ import annotations

import warnings

import numba as nb
import numpy as np
import pandas as pd


def create_batches(centroids: dict[str, np.ndarray], key: str = "batches") -> pd.Series:
    """Create batches."""
    batches = np.concatenate([np.full(len(v), k) for (k, v) in centroids.items()])
    return pd.Series(batches, name=key)


def combine_arrays(centroids: dict[str, np.ndarray]) -> np.ndarray:
    """Combine arrays."""
    return np.concatenate(list(centroids.values()))


def _one_hot_encode(
    labels: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Encode batch labels as a one-hot matrix using numpy (no pandas).

    Parameters
    ----------
    labels : np.ndarray
        1-D array of batch labels (any dtype).

    Returns
    -------
    unique_levels : (n_batches,) ndarray of sorted unique labels.
    batch_indices : list of index arrays, one per batch (sorted order).
    design : (n_obs, n_batches) float64 one-hot matrix.
    """
    unique_levels, inverse = np.unique(labels, return_inverse=True)
    n_batches = len(unique_levels)
    n_obs = len(labels)
    design = np.zeros((n_obs, n_batches), dtype=np.float64)
    batch_indices: list[np.ndarray] = []
    for j in range(n_batches):
        idx = np.where(inverse == j)[0]
        batch_indices.append(idx)
        design[idx, j] = 1.0
    return unique_levels, batch_indices, design


def compute_init_values_parametric(
    Z: np.ndarray, batches_one_hot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the starting values of the Bayesian optimization.

    Vectorized: ``gamma_hat`` is computed via a single matrix multiply instead
    of a Python loop; ``delta_hat_squared`` uses the algebraic identity
    ``Var(X) = E[X²] - E[X]²`` with Bessel correction.
    """
    # n_batches = batches_one_hot.shape[1]
    n_per_batch = batches_one_hot.sum(axis=0).astype(np.float64)  # (n_batches,)
    B = batches_one_hot.astype(np.float64)  # (n_obs, n_batches)

    # gamma_hat[i] = mean of Z rows belonging to batch i
    gamma_hat = (B.T @ Z) / n_per_batch[:, None]  # (n_batches, n_genes)

    gamma_bar = gamma_hat.mean(axis=1)  # (n_batches,)
    tau_bar_squared = gamma_hat.var(axis=1, ddof=1)  # (n_batches,)

    # delta_hat_squared[i] = Var(Z[batch_i], ddof=1) per gene
    # = (E[Z²] - E[Z]²) * n_i/(n_i-1)  via algebraic identity
    sum_sq = B.T @ (Z**2)  # (n_batches, n_genes)
    delta_hat_squared = (sum_sq / n_per_batch[:, None] - gamma_hat**2) * (
        n_per_batch[:, None] / (n_per_batch[:, None] - 1)
    )

    V_bar = delta_hat_squared.mean(axis=1)  # (n_batches,)
    S_bar_squared = delta_hat_squared.var(axis=1, ddof=1)  # (n_batches,)

    lambda_bar = (V_bar**2 + 2 * S_bar_squared) / S_bar_squared
    theta_bar = (V_bar**3 + V_bar * S_bar_squared) / S_bar_squared

    return gamma_hat, gamma_bar, delta_hat_squared, tau_bar_squared, lambda_bar, theta_bar


def compute_values_non_parametric(Z: np.ndarray, batches_one_hot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the starting values of the non-parametric Bayesian optimization.

    Vectorized via matrix multiply for ``gamma_hat`` and algebraic variance
    identity for ``delta_hat_squared``.
    """
    n_per_batch = batches_one_hot.sum(axis=0).astype(np.float64)  # (n_batches,)
    B = batches_one_hot.astype(np.float64)  # (n_obs, n_batches)

    gamma_hat = (B.T @ Z) / n_per_batch[:, None]  # (n_batches, n_genes)

    sum_sq = B.T @ (Z**2)  # (n_batches, n_genes)
    delta_hat_squared = (sum_sq / n_per_batch[:, None] - gamma_hat**2) * (
        n_per_batch[:, None] / (n_per_batch[:, None] - 1)
    )
    return gamma_hat, delta_hat_squared


def compute_weights(Z_i: np.ndarray, gamma_hat_i: np.ndarray, delta_hat_squared_i: np.ndarray, n_jobs: int = 1):
    """Compute the weights w_{ig} of the non-parametric Bayesian optimization."""
    from mpire import WorkerPool

    with WorkerPool(n_jobs=n_jobs) as pool:
        args = [(g, Z_i[:, g], gamma_hat_i, delta_hat_squared_i, Z_i.shape[1]) for g in range(Z_i.shape[1])]
        out = pool.map(
            compute_weights_util,
            args,
            progress_bar=True,
            progress_bar_options={"desc": "Computing weights...", "mininterval": 5},
        )
    return np.vstack(out)


def compute_weights_util(
    g: int, Z_ig: np.ndarray, gamma_hat_i: np.ndarray, delta_hat_squared_i: np.ndarray, n_genes: int
) -> np.ndarray:
    """Compute the weights w_{ig} of the non-parametric Bayesian optimization."""
    # Always delete the current gene as the parameters are confounded.
    tmp = (Z_ig.reshape(-1, 1) - np.delete(gamma_hat_i, g)) ** 2 / np.delete(delta_hat_squared_i, g)
    return np.prod((1 / np.sqrt(2 * np.pi * np.delete(delta_hat_squared_i, g))) * np.exp(-0.5 * tmp), axis=0)


def new_gamma_parametric(n_i, tau_bar_squared_i, gamma_hat_i, delta_star_squared_i, gamma_bar_i):
    """Compute gamma star."""
    numerator = n_i * tau_bar_squared_i * gamma_hat_i + delta_star_squared_i * gamma_bar_i
    denominator = n_i * tau_bar_squared_i + delta_star_squared_i
    return numerator / denominator


def new_delta_star_squared_parametric(theta_bar_i, Z_i, gamma_star_new_i, n_i, lambda_bar_i):
    """Compute delta star."""
    numerator = theta_bar_i + 0.5 * np.sum((Z_i - gamma_star_new_i) ** 2, axis=0)
    denominator = 0.5 * n_i + lambda_bar_i - 1
    return numerator / denominator


@nb.njit(fastmath=True, cache=True)
def _nb_parametric_update(
    gamma_star_i: np.ndarray,
    delta_star_squared_i: np.ndarray,
    n_i: float,
    tau_bar_squared_i: float,
    gamma_hat_i: np.ndarray,
    gamma_bar_i: float,
    theta_bar_i: float,
    lambda_bar_i: float,
    Z_i: np.ndarray,
    conv_criterion: float,
    max_iter: int,
) -> tuple:
    """Numba JIT convergence loop for parametric empirical Bayes optimization.

    ``Z_i`` is ``(n_samples, n_genes)`` C-contiguous.  The inner loops access
    rows of ``Z_i`` sequentially, maximising cache reuse.

    Returns ``(gamma_star_i, delta_star_squared_i, hit_max_iter)``.
    """
    n_genes = gamma_hat_i.shape[0]
    n_samples = Z_i.shape[0]

    gamma_star_new = gamma_star_i.copy()
    delta_star_squared_new = delta_star_squared_i.copy()

    tau_n = n_i * tau_bar_squared_i  # scalar, precomputed outside inner loop
    half_n_denom = 0.5 * n_i + lambda_bar_i - 1.0

    iterations = 0
    convergence = conv_criterion + 1.0

    while convergence > conv_criterion and iterations < max_iter:
        # --- update gamma (new_gamma_parametric inlined) ---
        for g in range(n_genes):
            num = tau_n * gamma_hat_i[g] + delta_star_squared_i[g] * gamma_bar_i
            den = tau_n + delta_star_squared_i[g]
            gamma_star_new[g] = num / den

        # --- update delta (new_delta_star_squared_parametric inlined) ---
        # Access Z_i row-by-row (cache-friendly for C-contiguous layout)
        sum_sq = np.zeros(n_genes)
        for s in range(n_samples):
            for g in range(n_genes):
                d = Z_i[s, g] - gamma_star_new[g]
                sum_sq[g] += d * d
        for g in range(n_genes):
            delta_star_squared_new[g] = (theta_bar_i + 0.5 * sum_sq[g]) / half_n_denom

        # --- convergence: max relative change (matches original formula) ---
        c_g = 0.0
        for g in range(n_genes):
            v = abs(gamma_star_new[g] - gamma_star_i[g]) / gamma_star_i[g]
            if v > c_g:
                c_g = v
        c_d = 0.0
        for g in range(n_genes):
            v = abs(delta_star_squared_new[g] - delta_star_squared_i[g]) / delta_star_squared_i[g]
            if v > c_d:
                c_d = v
        convergence = c_g if c_g > c_d else c_d

        gamma_star_i[:] = gamma_star_new
        delta_star_squared_i[:] = delta_star_squared_new
        iterations += 1

    return gamma_star_i, delta_star_squared_i, iterations >= max_iter


def parametric_update(
    gamma_star_i: np.ndarray,
    delta_star_squared_i: np.ndarray,
    n_i,
    tau_bar_squared_i,
    gamma_hat_i: np.ndarray,
    gamma_bar_i,
    theta_bar_i,
    lambda_bar_i,
    Z_i: np.ndarray,
    conv_criterion: float = 1e-4,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform the parametric EB optimization for one batch.

    Dispatches to a numba JIT inner loop for performance.
    """
    g_out, d_out, hit_max = _nb_parametric_update(
        np.asarray(gamma_star_i, dtype=np.float64),
        np.asarray(delta_star_squared_i, dtype=np.float64),
        float(n_i),
        float(tau_bar_squared_i),
        np.asarray(gamma_hat_i, dtype=np.float64),
        float(gamma_bar_i),
        float(theta_bar_i),
        float(lambda_bar_i),
        np.ascontiguousarray(Z_i, dtype=np.float64),
        float(conv_criterion),
        int(max_iter),
    )
    if hit_max:
        warnings.warn("Maximum number of iterations reached", RuntimeWarning, stacklevel=2)
    return g_out, d_out
