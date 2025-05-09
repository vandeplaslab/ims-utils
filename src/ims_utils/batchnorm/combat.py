"""Batch correction methods."""

from __future__ import annotations

import typing as ty

import numpy as np
import pandas as pd
from koyo.timer import MeasureTimer
from loguru import logger
from numpy import linalg as la
from scipy.sparse import issparse


class ComBat:
    """ComBat algorithm."""

    def __init__(self, key: str = "batches"):
        self.key = key

    def fit_transform(self, array: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return combat(array, batches, key=self.key)


def combat(
    array: np.ndarray,
    batches: np.ndarray | pd.Series,
    key: str = "batches",
) -> np.ndarray | np.ndarray | None:
    """
    ComBat function for batch effect correction [Johnson07]_ [Leek12]_
    [Pedersen12]_.
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
    # only works on dense matrices so far
    data = pd.DataFrame(data=array.toarray().T if issparse(array) else array.T)  # type: ignore

    # construct a pandas series of the batch annotation
    if isinstance(batches, np.ndarray):
        batches = pd.Series(batches, name=key)
    batches = pd.DataFrame(batches, columns=[key])
    batch_info = batches.groupby(key).indices.values()
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # standardize across genes using a pooled variance estimator
    with MeasureTimer(func=logger.debug, msg="Standardized data {}"):
        s_data, design, var_pooled, stand_mean = _standardize_data(batches, data, key)
        # cleanup
        del batches, data

    # fitting the parameters on the standardized data
    with MeasureTimer(func=logger.debug, msg="Fitted model in {}"):
        batch_design = design[design.columns[:n_batch]]
        # first estimate of the additive batch effect
        gamma_hat = (la.inv(batch_design.T @ batch_design) @ batch_design.T @ s_data.T).values
        delta_hat = []

        # first estimate for the multiplicative batch effect
        for i, batch_index in enumerate(batch_info):
            delta_hat.append(s_data.iloc[:, batch_index].var(axis=1))

        # empirically fix the prior hyperparameters
        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1)
        # a_prior and b_prior are the priors on lambda and theta from Johnson and Li (2006)
        a_prior = list(map(_aprior, delta_hat))
        b_prior = list(map(_bprior, delta_hat))

        # gamma star and delta star will be our empirical bayes (EB) estimators
        # for the additive and multiplicative batch effect per batch and cell
        gamma_star, delta_star = [], []
        for i, batch_index in enumerate(batch_info):
            # temp stores our estimates for the batch effect parameters.
            # temp[0] is the additive batch effect
            # temp[1] is the multiplicative batch effect
            gamma, delta = _it_sol(
                s_data.iloc[:, batch_index].values,
                gamma_hat[i],
                delta_hat[i].values,
                gamma_bar[i],
                t2[i],
                a_prior[i],
                b_prior[i],
            )

            gamma_star.append(gamma)
            delta_star.append(delta)
        # cleanup
        del gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior

    with MeasureTimer(func=logger.debug, msg="Corrected data in {}"):
        bayes_data = s_data
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)

        # we now apply the parametric adjustment to the standardized data from above
        # loop over all batches in the data
        for j, batch_index in enumerate(batch_info):
            # we basically subtract the additive batch effect, rescale by the ratio
            # of multiplicative batch effect to pooled variance and add the overall gene
            # wise mean
            dsq = np.sqrt(delta_star[j, :])
            dsq = dsq.reshape((len(dsq), 1))
            denominator = np.dot(dsq, np.ones((1, n_batches[j])))
            value = np.array(bayes_data.iloc[:, batch_index] - np.dot(batch_design.iloc[batch_index], gamma_star).T)
            bayes_data.iloc[:, batch_index] = value / denominator

        vp_sq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        bayes_data = bayes_data * np.dot(vp_sq, np.ones((1, int(n_array)))) + stand_mean
        # cleanup
        del vp_sq, stand_mean, var_pooled, design, s_data, gamma_star, delta_star
    return bayes_data.values.transpose()


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
    """\
    Iteratively compute the conditional posterior means for gamma and delta.

    gamma is an estimator for the additive batch effect, deltat is an estimator
    for the multiplicative batch effect. We use an EB framework to estimate these
    two. Analytical expressions exist for both parameters, which however depend on each other.
    We therefore iteratively evaluate these two expressions until convergence is reached.

    Parameters
    ----------
    s_data
        Contains the standardized Data
    g_hat
        Initial guess for gamma
    d_hat
        Initial guess for delta
    g_bar, t2, a, b
        Hyperparameters
    conv: float, optional (default: `0.0001`)
        convergence criterium

    Returns
    -------
    g_new
        estimated value for gamma
    d_new
        estimated value for delta
    """
    n = (1 - np.isnan(s_data)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change: float = 1.0
    count: int = 0

    # They need to be initialized for numba to properly infer types
    g_new = g_old
    d_new = d_old
    # we place a normally distributed prior on gamma and inverse gamma prior on delta
    # in the loop, gamma and delta are updated together. they depend on each other. we iterate until convergence.
    while change > conv:
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        sum2 = s_data - g_new.reshape((g_new.shape[0], 1)) @ np.ones((1, s_data.shape[1]))
        sum2 = sum2**2
        sum2 = sum2.sum(axis=1)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new  # .copy()
        d_old = d_new  # .copy()
        count = count + 1
    return g_new, d_new


def _aprior(delta_hat: np.ndarray):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2 * s2 + m**2) / s2


def _bprior(delta_hat: np.ndarray):
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m**3) / s2


def _standardize_data(
    model: pd.DataFrame, data: pd.DataFrame, batch_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """\
    Standardizes the data per gene.

    The aim here is to make mean and variance be comparable across batches.

    Parameters
    ----------
    model
        Contains the batch annotation
    data
        Contains the Data
    batch_key
        Name of the batch column in the model matrix

    Returns
    -------
    s_data
        Standardized Data
    design
        Batch assignment as one-hot encodings
    var_pooled
        Pooled variance per gene
    stand_mean
        Gene-wise mean
    """
    # compute the design matrix
    batch_items = model.groupby(batch_key).groups.items()
    batch_levels, batch_info = zip(*batch_items)
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # design = pd.get_dummies(model.astype(str))
    design = _design_matrix(model, batch_key, batch_levels)

    # compute pooled variance estimator
    b_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, b_hat[:n_batch, :])
    var_pooled = (data - np.dot(design, b_hat).T) ** 2
    var_pooled = np.dot(var_pooled, np.ones((int(n_array), 1)) / int(n_array))

    # Compute the means
    if np.sum(var_pooled == 0) > 0:
        logger.warning(f"Found {np.sum(var_pooled == 0)} genes with zero variance.")
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, b_hat).T

    # need to be a bit careful with the zero variance genes
    # just set the zero variance genes to zero in the standardized data
    s_data = np.where(
        var_pooled == 0,
        0,
        ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array))))),
    )
    s_data = pd.DataFrame(s_data, index=data.index, columns=data.columns)

    return s_data, design, var_pooled, stand_mean


# noinspection PyUnusedLocal
def _design_matrix(model: pd.DataFrame, batch_key: str, batch_levels: ty.Collection[str]) -> pd.DataFrame:
    """\
    Computes a simple design matrix.

    Parameters
    ----------
    model
        Contains the batch annotation
    batch_key
        Name of the batch column

    Returns
    -------
    The design matrix for the regression problem
    """
    import patsy

    design = patsy.dmatrix(  # type: ignore
        f"~ 0 + C(Q('{batch_key}'), levels=batch_levels)",
        model,
        return_type="dataframe",
    )
    model = model.drop([batch_key], axis=1)
    numerical_covariates = model.select_dtypes("number").columns.values

    logger.debug(f"Found {design.shape[1]} batches")
    other_cols = [c for c in model.columns.values if c not in numerical_covariates]

    if other_cols:
        col_repr = " + ".join(f"Q('{x}')" for x in other_cols)
        factor_matrix = patsy.dmatrix(  # type: ignore
            f"~ 0 + {col_repr}",
            model[other_cols],
            return_type="dataframe",
        )

        design = pd.concat((design, factor_matrix), axis=1)
        logger.info(f"Found {len(other_cols)} categorical variables: {', '.join(other_cols)}")

    if numerical_covariates is not None:
        logger.info(f"Found {len(numerical_covariates)} numerical variables: {', '.join(numerical_covariates)}")

        for nC in numerical_covariates:
            design[nC] = model[nC]

    return design
