"""
reComBat.

The main class of the reComBat algorithm.

Copied and modified from:
https://github.com/BorgwardtLab/reComBat

Original license:
BSD 3-Clause License

# Author: Michael F. Adamer <mikeadamer@gmail.com>
# November 2021
"""

from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from koyo.timer import MeasureTimer
from koyo.utilities import is_installed
from loguru import logger
from tqdm import tqdm

from ims_utils.batchnorm.utilities import (
    compute_init_values_parametric,
    compute_values_non_parametric,
    compute_weights,
    parametric_update,
)

if not is_installed("patsy"):
    raise ImportError("scikit-learn is required for this module.")


def recombat(array: np.ndarray, batches: np.ndarray, key: str = "batch", **kwargs):
    """Calculate recombat."""
    combat = ReComBat(**kwargs)
    return combat.fit_transform(pd.DataFrame(array), pd.Series(batches, name=key)).values


class ReComBat:
    """reComBat algorithm.

    Parameters
    ----------
    parametric : bool, optional
        Choose a parametric or non-parametric empirical Bayes optimization.
        The default is True.
    model : str, optional
        Choose a linear model, ridge, Lasso, elastic_net.
        The default is 'linear'.
    config : dict, optional
        A dictionary containing kwargs for the model (see sklean.linear_model for details).
        The default is None.
    conv_criterion : float, optional
        The convergence criterion for the optimization.
        The default is 1e-4.
    max_iter : int, optional
        The maximum number of steps of the parametric empirical Bayes optimization.
        The detault is 1000.
    n_jobs : int, optional
        The number of parallel threads in the non-parametric optimization.
        If not given, then this is set to the number of cpus.
    mean_only : bool, optional
        Adjust the mean only. No scaling is performed.
        The default is False.
    optimize_params : bool, optional
        Perform empirical Bayes optimization.
        The default is True.
    reference_batch : str, optional
        Give a reference batch which is not adjusted.
        The default is None.
    """

    alpha_ = None
    beta_x_ = None
    beta_c_ = None
    sigma_ = None
    gamma_star_hat_ = None
    delta_star_squared_hat_ = None

    def __init__(
        self,
        parametric=True,
        model="elastic_net",
        config=None,
        conv_criterion=1e-4,
        max_iter=1000,
        n_jobs=None,
        mean_only=False,
        optimize_params=True,
        reference_batch=None,
    ):
        self.parametric = parametric
        self.model = model
        if config is not None:
            self.config = config
        else:
            self.config = {}
        self.conv_criterion = conv_criterion
        self.max_iter = max_iter
        if n_jobs is None:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        self.mean_only = mean_only
        self.optimize_params = optimize_params
        if reference_batch is not None:
            self.reference_batch = reference_batch
        else:
            self.reference_batch = None

        # Set to True to indicate that reComBat has been fitted.
        self.is_fitted = False

    def fit(self, data, batches):
        """
        Fit method.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe containing the data matrix.
            The format is (rows x columns) = (samples x features)
        batches : pandas series
            A pandas series containing the batch of each sample in the dataframe.

        Returns
        -------
        None
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(batches, np.ndarray):
            batches = pd.Series(batches, name="batches")

        with MeasureTimer(func=logger.debug, msg="Fitted reComBat in {}"):
            if data.isna().any().any():
                raise ValueError("The data contains NaN values.")
            if batches.isna().any().any():
                raise ValueError("The batches contain NaN values.")

            unique_batches = batches.unique()
            num_batches = len(unique_batches)

            if num_batches == 1:
                raise ValueError("There should be at least two batches in the dataset.")

            batches_one_hot = pd.get_dummies(batches.astype(str)).values

            if np.any(batches_one_hot.sum(axis=0) == 1):
                raise ValueError("There should be at least two values for each batch.")

            if data.shape[0] != batches.shape[0]:
                raise ValueError("The batch and data matrix have a different number of samples.")

            with MeasureTimer(func=logger.debug, msg="Fit linear model in {}"):
                array = self._fit_model(data, batches_one_hot)

            if self.optimize_params:
                if self.parametric:
                    with MeasureTimer(func=logger.debug, msg="Fit parametric model in {}"):
                        self._parametric_optimization(array, batches_one_hot)
                elif not self.parametric:
                    with MeasureTimer(func=logger.debug, msg="Fit non-parametric model in {}"):
                        self._non_parametric_optimization(array, batches_one_hot)
            else:
                with MeasureTimer(func=logger.debug, msg="Computed parameters without optimization in {}"):
                    self.gamma_star_hat_, self.delta_star_squared_hat_ = compute_values_non_parametric(
                        array, batches_one_hot
                    )
            self.is_fitted = True

    def transform(self, data: pd.DataFrame, batches):
        """
        Transform method.
        ----------------.

        Adjusts a dataframe. Please make sure that the number of batches,
        features and design matrix features match.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe containing the data matrix.
            The format is (rows x columns) = (samples x features)
        batches : pandas series
            A pandas series containing the batch of each sample in the dataframe.

        Returns
        -------
        A pandas dataframe of the same shape as the input dataframe.
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if isinstance(batches, np.ndarray):
            batches = pd.Series(batches, name="batches")

        with MeasureTimer(func=logger.debug, msg="Transformed reComBat in {}"):
            batches_one_hot = pd.get_dummies(batches.astype(str)).values
            if not self.is_fitted:
                raise AttributeError("reComBat has not been fitted yet.")
            if data.shape[1] != self.alpha_.shape[1]:
                raise ValueError("Wrong number of features.")
            if batches_one_hot.shape[1] != self.gamma_star_hat_.shape[0]:
                raise ValueError("Wrong number of batches.")

            array = (data.values.copy() - self.alpha_) / np.sqrt(self.sigma_)
        return pd.DataFrame(self._adjust_data(array, batches_one_hot), index=data.index, columns=data.columns)

    def fit_transform(self, data: pd.DataFrame, batches):
        """Fit and transform in one go."""
        self.fit(data, batches)
        return self.transform(data, batches)

    def _fit_model(self, data, batches_one_hot):
        """Fit the linear model."""
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

        num_batches = batches_one_hot.shape[1]
        cov = batches_one_hot
        x_cov_dim = 0

        # Initialise the model class
        if self.model == "linear":
            model = LinearRegression(fit_intercept=False, **self.config)
        elif self.model == "ridge":
            model = Ridge(fit_intercept=False, **self.config)
        elif self.model == "Lasso":
            model = Lasso(fit_intercept=False, **self.config)
        elif self.model == "elastic_net":
            model = ElasticNet(fit_intercept=False, **self.config)
        else:
            raise ValueError("Model not implemented")

        model.fit(cov, data.values)

        # Save the fitted parameters
        # Note that alpha is computed implicitly via the constraints on the batch parameters.
        self.alpha_ = np.matmul(
            batches_one_hot.sum(axis=0, keepdims=True) / batches_one_hot.sum(), model.coef_.T[:num_batches]
        )
        self.beta_x_ = model.coef_.T[num_batches : num_batches + x_cov_dim]
        self.beta_c_ = model.coef_.T[num_batches + x_cov_dim :]

        # Compute the standard deviation of the reconstructed data.
        data_hat = np.matmul(cov, model.coef_.T)
        self.sigma_ = np.mean((data.values - data_hat) ** 2, axis=0, keepdims=True)

        # Standardise the data.
        array = (
            data.values.copy()
            - np.matmul(cov[:, num_batches : num_batches + x_cov_dim], self.beta_x_)
            - np.matmul(cov[:, num_batches + x_cov_dim :], self.beta_c_)
            - self.alpha_
        ) / np.sqrt(self.sigma_)
        return array

    def _parametric_optimization(self, array, batches_one_hot):
        """Perform parametric optimization."""
        (
            gamma_hat,
            gamma_bar,
            delta_hat_squared,
            tau_bar_squared,
            lambda_bar,
            theta_bar,
        ) = compute_init_values_parametric(array, batches_one_hot)
        n = batches_one_hot.sum(axis=0)

        gamma_star = gamma_hat.copy()
        delta_star_squared = delta_hat_squared.copy()

        for i in range(batches_one_hot.shape[1]):
            if not self.mean_only:
                gamma_star[i], delta_star_squared[i] = parametric_update(
                    gamma_star[i],
                    delta_star_squared[i],
                    n[i],
                    tau_bar_squared[i],
                    gamma_hat[i],
                    gamma_bar[i],
                    theta_bar[i],
                    lambda_bar[i],
                    array[batches_one_hot[:, i] == 1],
                    conv_criterion=self.conv_criterion,
                    max_iter=self.max_iter,
                )
            else:
                gamma_star[i] = self._new_gamma_parametric(1, tau_bar_squared[i], gamma_hat[i], 1, gamma_bar[i])
                delta_star_squared[i] = np.ones_like(delta_hat_squared[i])
        self.gamma_star_hat_ = gamma_star
        self.delta_star_squared_hat_ = delta_star_squared

    @staticmethod
    def _new_gamma_parametric(n_i, tau_bar_squared_i, gamma_hat_i, delta_star_squared_i, gamma_bar_i):
        """New gamma."""
        numerator = n_i * tau_bar_squared_i * gamma_hat_i + delta_star_squared_i * gamma_bar_i
        denominator = n_i * tau_bar_squared_i + delta_star_squared_i
        return numerator / denominator

    def _non_parametric_optimization(self, array, batches_one_hot):
        """Perform non-parametric optimization."""
        gamma_hat, delta_hat_squared = compute_values_non_parametric(array, batches_one_hot)

        gamma_star = np.zeros((batches_one_hot.shape[1], array.shape[1]))
        delta_star_squared = np.zeros((batches_one_hot.shape[1], array.shape[1]))
        for i in tqdm(range(batches_one_hot.shape[1])):
            if self.mean_only:
                delta_hat_squared[i] = np.ones_like(delta_hat_squared[i])
            weights = compute_weights(
                array[batches_one_hot[:, i] == 1], gamma_hat[i], delta_hat_squared[i], n_jobs=self.n_jobs
            )

            gamma_star_numerator = np.vstack([weights[j] * np.delete(gamma_hat[i], j) for j in range(array.shape[1])])
            gamma_star[i] = (np.sum(gamma_star_numerator, axis=1) / np.sum(weights, axis=1)).T

            delta_star_numerator = np.vstack(
                [weights[j] * np.delete(delta_hat_squared[i], j) for j in range(array.shape[1])]
            )
            delta_star_squared[i] = (np.sum(delta_star_numerator, axis=1) / np.sum(weights, axis=1)).T

        self.gamma_star_hat_ = gamma_star
        self.delta_star_squared_hat_ = delta_star_squared

    def _adjust_data(self, array, batches_one_hot):
        """Perform the final adjustment step."""
        tmp = np.zeros_like(array, dtype=np.float32)
        for i in range(batches_one_hot.shape[1]):
            tmp[batches_one_hot[:, i] == 1] = (array[batches_one_hot[:, i] == 1] - self.gamma_star_hat_[i]) / np.sqrt(
                self.delta_star_squared_hat_[i]
            )
        tmp = np.sqrt(self.sigma_.astype(np.float32)) * tmp + self.alpha_.astype(np.float32)
        return tmp.astype(np.float32)
