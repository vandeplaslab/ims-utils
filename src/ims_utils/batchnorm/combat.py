"""Batch correction methods."""

from __future__ import annotations

import typing as ty

import numba as nb
import numpy as np
from koyo.timer import MeasureTimer
from loguru import logger
from numpy import linalg as la
from scipy.sparse import issparse

from ims_utils.batchnorm.utilities import _one_hot_encode

if ty.TYPE_CHECKING:
    import pandas as pd


class ComBat:
    """ComBat algorithm."""

    def __init__(self, key: str = "batches"):
        self.key = key

    def fit_transform(self, array: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return combat(array, batches, key=self.key)


@nb.njit(fastmath=True, cache=True)
def _nb_it_sol(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float,
) -> tuple:
    """Numba JIT empirical Bayes iteration (inner loop of :func:`_it_sol`).

    Parameters
    ----------
    s_data : float64[n_genes, n_batch_samples]
        Standardized data for one batch (genes x samples).
    g_hat, d_hat : float64[n_genes]
        Initial estimates for additive / multiplicative batch effect.
    g_bar, t2, a, b : float
        Hyperparameters.
    conv : float
        Convergence criterion.

    Returns
    -------
    (g_new, d_new)
    """
    n_genes = s_data.shape[0]
    n_samples = s_data.shape[1]

    # Count non-NaN samples per gene (row-wise, cache-friendly for row-major)
    n = np.empty(n_genes)
    for i in range(n_genes):
        c = 0
        for j in range(n_samples):
            if not np.isnan(s_data[i, j]):
                c += 1
        n[i] = float(c)

    tn = t2 * n  # t2 * n[i], precomputed

    g_old = g_hat.copy()
    d_old = d_hat.copy()
    g_new = g_old.copy()
    d_new = d_old.copy()

    change = 1.0
    while change > conv:
        # Update gamma
        for i in range(n_genes):
            g_new[i] = (tn[i] * g_hat[i] + d_old[i] * g_bar) / (tn[i] + d_old[i])

        # Update delta — inner loop over samples is cache-friendly (row-major)
        for i in range(n_genes):
            ss = 0.0
            gi = g_new[i]
            for j in range(n_samples):
                d = s_data[i, j] - gi
                ss += d * d
            d_new[i] = (0.5 * ss + b) / (n[i] / 2.0 + a - 1.0)

        # Convergence: max relative change (matches original formula)
        c1 = 0.0
        for i in range(n_genes):
            v = abs(g_new[i] - g_old[i]) / g_old[i]
            if v > c1:
                c1 = v
        c2 = 0.0
        for i in range(n_genes):
            v = abs(d_new[i] - d_old[i]) / d_old[i]
            if v > c2:
                c2 = v
        change = c1 if c1 > c2 else c2

        g_old[:] = g_new
        d_old[:] = d_new

    return g_new, d_new


def _it_sol(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively compute the conditional posterior means for gamma and delta.

    Dispatches to a numba JIT inner loop.  See :func:`_nb_it_sol` for the
    algorithmic details.

    Parameters
    ----------
    s_data
        Contains the standardized data (n_genes x n_batch_samples).
    g_hat
        Initial guess for gamma.
    d_hat
        Initial guess for delta.
    g_bar, t2, a, b
        Hyperparameters.
    conv : float, optional (default: ``0.0001``)
        Convergence criterion.

    Returns
    -------
    g_new
        Estimated value for gamma.
    d_new
        Estimated value for delta.
    """
    return _nb_it_sol(
        np.ascontiguousarray(s_data, dtype=np.float64),
        np.asarray(g_hat, dtype=np.float64),
        np.asarray(d_hat, dtype=np.float64),
        float(g_bar),
        float(t2),
        float(a),
        float(b),
        float(conv),
    )


def _aprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2 * s2 + m**2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m**3) / s2


def _standardize_data(
    design: np.ndarray,
    data: np.ndarray,
    n_batch: int,
    n_batches: np.ndarray,
    n_array: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize data per gene.

    All operations are in numpy — no pandas DataFrame involved.

    Parameters
    ----------
    design : float64[n_obs, n_design_cols]
        Full design matrix (batch one-hot columns first).
    data : float64[n_genes, n_obs]
        Raw data in ComBat's genes x samples orientation.
    n_batch : int
        Number of batch columns in ``design``.
    n_batches : int[n_batch]
        Number of samples per batch.
    n_array : float
        Total number of samples.

    Returns
    -------
    s_data : float64[n_genes, n_obs]
        Standardized data.
    var_pooled : float64[n_genes, 1]
        Pooled per-gene variance.
    stand_mean : float64[n_genes, n_obs]
        Per-gene grand-mean broadcast across samples.
    """
    # Solve for the regression coefficients: b_hat[n_design_cols, n_genes]
    b_hat = la.solve(design.T @ design, design.T @ data.T)

    grand_mean = (n_batches / n_array) @ b_hat[:n_batch, :]  # (n_genes,)

    # Pooled variance: mean squared residual across all samples
    fitted = design @ b_hat  # (n_obs, n_genes)
    residuals = data.T - fitted  # (n_obs, n_genes)
    var_pooled = (residuals**2).mean(axis=0, keepdims=True).T  # (n_genes, 1)

    if np.any(var_pooled == 0):
        logger.warning(f"Found {np.sum(var_pooled == 0)} genes with zero variance.")

    # Stand mean: grand mean + covariate contribution (batch cols zeroed out)
    stand_mean = np.outer(grand_mean, np.ones(int(n_array)))  # (n_genes, n_obs)
    if design.shape[1] > n_batch:
        tmp = design.copy()
        tmp[:, :n_batch] = 0.0
        stand_mean += (tmp @ b_hat).T

    # Standardize
    sqrt_var = np.sqrt(var_pooled)  # (n_genes, 1)
    s_data = np.where(var_pooled == 0, 0.0, (data - stand_mean) / sqrt_var)

    return s_data, var_pooled, stand_mean


def combat(
    array: np.ndarray,
    batches: np.ndarray | pd.Series,
    key: str = "batches",
) -> np.ndarray:
    """ComBat function for batch effect correction [Johnson07]_ [Leek12]_ [Pedersen12]_.

    Corrects for batch effects by fitting linear models, gains statistical power
    via an EB framework where information is borrowed across genes.
    This uses the implementation `combat.py`_ [Pedersen12]_.
    .. _combat.py: https://github.com/brentp/combat.py.

    Parameters
    ----------
    array : np.ndarray
        Array with [observations, features] shape.
    batches : np.ndarray
        Array with batch information where each row corresponds to a sample index.
    key : str, optional
        Key of batch information.

    Returns
    -------
    np.ndarray
        Corrected array.
    """
    # Work in ComBat's genes x samples orientation
    X = array.toarray().T if issparse(array) else array.T  # (n_genes, n_obs)
    X = np.asarray(X, dtype=np.float64)

    # Normalise batch labels to a string array for consistent ordering
    try:
        import pandas as pd

        if isinstance(batches, (pd.Series, pd.DataFrame)):
            batch_labels = batches.values.ravel().astype(str)
        else:
            batch_labels = np.asarray(batches).astype(str)
    except ImportError:
        batch_labels = np.asarray(batches).astype(str)

    _, batch_info, design = _one_hot_encode(batch_labels)
    n_batch = design.shape[1]
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(X.shape[1])

    # Standardize across genes using a pooled variance estimator
    with MeasureTimer(func=logger.debug, msg="Standardized data {}"):
        s_data, var_pooled, stand_mean = _standardize_data(design, X, n_batch, n_batches, n_array)

    # Fit parameters on the standardized data
    with MeasureTimer(func=logger.debug, msg="Fitted model in {}"):
        batch_design = design[:, :n_batch]  # (n_obs, n_batches)

        # First estimate of the additive batch effect: (n_batches, n_genes)
        gamma_hat = la.solve(batch_design.T @ batch_design, batch_design.T @ s_data.T)

        # First estimate for the multiplicative batch effect per batch
        delta_hat = [s_data[:, bi].var(axis=1) for bi in batch_info]

        # Empirically fix the prior hyperparameters
        gamma_bar = gamma_hat.mean(axis=1)  # (n_batches,)
        t2 = gamma_hat.var(axis=1)  # (n_batches,)
        a_prior = list(map(_aprior, delta_hat))
        b_prior = list(map(_bprior, delta_hat))

        # EB estimators for the additive and multiplicative batch effects
        gamma_star, delta_star = [], []
        for i, bi in enumerate(batch_info):
            gamma, delta = _it_sol(
                s_data[:, bi],
                gamma_hat[i],
                delta_hat[i],
                gamma_bar[i],
                t2[i],
                a_prior[i],
                b_prior[i],
            )
            gamma_star.append(gamma)
            delta_star.append(delta)

        del gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior

    with MeasureTimer(func=logger.debug, msg="Corrected data in {}"):
        gamma_star = np.array(gamma_star)  # (n_batches, n_genes)
        delta_star = np.array(delta_star)  # (n_batches, n_genes)

        # Apply the parametric correction batch by batch
        for j, bi in enumerate(batch_info):
            dsq = np.sqrt(delta_star[j, :, np.newaxis])  # (n_genes, 1)
            gamma_contrib = (batch_design[bi] @ gamma_star).T  # (n_genes, n_batch)
            s_data[:, bi] = (s_data[:, bi] - gamma_contrib) / dsq

        s_data = s_data * np.sqrt(var_pooled) + stand_mean
        del var_pooled, stand_mean, design, gamma_star, delta_star

    return s_data.T  # (n_obs, n_genes)
