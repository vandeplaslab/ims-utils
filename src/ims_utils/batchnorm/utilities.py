import typing as ty
import warnings

import numpy as np
import pandas as pd
from mpire import WorkerPool

if ty.TYPE_CHECKING:
    import pandas as pd


def create_batches(centroids: dict[str, np.ndarray], key: str = "batches") -> pd.Series:
    """Create batches."""
    batches = np.concatenate([np.full(len(v), k) for (k, v) in centroids.items()])
    return pd.Series(batches, name=key)


def combine_arrays(centroids: dict[str, np.ndarray]) -> np.ndarray:
    """Combine arrays."""
    return np.concatenate(list(centroids.values()))


def compute_init_values_parametric(Z, batches_one_hot):
    """Compute the starting values of the Bayesian optimization."""
    gamma_hat = np.array([np.mean(Z[batches_one_hot[:, i] == 1], axis=0) for i in range(batches_one_hot.shape[1])])
    gamma_bar = np.mean(gamma_hat, axis=1)
    tau_bar_squared = np.var(gamma_hat, axis=1, ddof=1)
    delta_hat_squared = np.array(
        [np.var(Z[batches_one_hot[:, i] == 1], axis=0, ddof=1) for i in range(batches_one_hot.shape[1])]
    )
    V_bar = np.mean(delta_hat_squared, axis=1)
    S_bar_squared = np.var(delta_hat_squared, axis=1, ddof=1)

    lambda_bar = (V_bar**2 + 2 * S_bar_squared) / S_bar_squared
    theta_bar = (V_bar**3 + V_bar * S_bar_squared) / S_bar_squared

    return gamma_hat, gamma_bar, delta_hat_squared, tau_bar_squared, lambda_bar, theta_bar


def compute_values_non_parametric(Z, batches_one_hot):
    """Compute the starting values of the Bayesian optimization."""
    gamma_hat = np.array([np.mean(Z[batches_one_hot[:, i] == 1], axis=0) for i in range(batches_one_hot.shape[1])])
    delta_hat_squared = np.array(
        [np.var(Z[batches_one_hot[:, i] == 1], axis=0, ddof=1) for i in range(batches_one_hot.shape[1])]
    )
    return gamma_hat, delta_hat_squared


def compute_weights(Z_i, gamma_hat_i, delta_hat_squared_i, n_jobs=1):
    """Compute the weights w_{ig} of the non-parametric Bayesian optimization."""
    with WorkerPool(n_jobs=n_jobs) as pool:
        args = [(g, Z_i[:, g], gamma_hat_i, delta_hat_squared_i, Z_i.shape[1]) for g in range(Z_i.shape[1])]
        out = pool.map(
            compute_weights_util,
            args,
            progress_bar=True,
            progress_bar_options={"desc": "Computing weights...", "mininterval": 5},
        )
    return np.vstack(out)


def compute_weights_util(g, Z_ig, gamma_hat_i, delta_hat_squared_i, n_genes):
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


def parametric_update(
    gamma_star_i,
    delta_star_squared_i,
    n_i,
    tau_bar_squared_i,
    gamma_hat_i,
    gamma_bar_i,
    theta_bar_i,
    lambda_bar_i,
    Z_i,
    conv_criterion=1e-4,
    max_iter=1000,
):
    """Perform the optimization for one batch at a time."""
    gamma_star_new_i = gamma_star_i.copy()
    delta_star_squared_new_i = delta_star_squared_i.copy()

    iterations = 0
    convergence = conv_criterion + 1
    while (convergence > conv_criterion) and (iterations < max_iter):
        gamma_star_new_i = new_gamma_parametric(n_i, tau_bar_squared_i, gamma_hat_i, delta_star_squared_i, gamma_bar_i)
        delta_star_squared_new_i = new_delta_star_squared_parametric(
            theta_bar_i, Z_i, gamma_star_new_i, n_i, lambda_bar_i
        )

        convergence = np.max(
            [
                np.max(np.abs(gamma_star_new_i - gamma_star_i) / gamma_star_i),
                np.max(np.abs(delta_star_squared_new_i - delta_star_squared_i) / delta_star_squared_i),
            ]
        )
        gamma_star_i = gamma_star_new_i
        delta_star_squared_i = delta_star_squared_new_i
        iterations += 1
    if iterations >= max_iter:
        warnings.warn("Maximum number of iterations reached", RuntimeWarning)
    return gamma_star_i, delta_star_squared_i
