# copyright: your name or organization
"""Deterministic growth forecaster."""

__author__ = ["Davi Lisboa Da Silva"]
__all__ = ["DeterministicGrowthForecaster"]

import numbers

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class DeterministicGrowthForecaster(BaseForecaster):
    """
    Deterministic forecaster with constant or time-varying percentage growth.

    This forecaster does not estimate a statistical model. It projects the last
    observed value of the target series using either:

    1. A single growth rate applied to all forecast periods.
    2. A sequence of growth rates, one for each forecast period.
    3. A shorter sequence of growth rates, where the last provided rate is
       repeated until the full forecast horizon is covered.

    Parameters
    ----------
    growth_rate : int, float, or list of int/float, default=0.02
        Growth rate assumption used to project the target series.

        If int or float, the same growth rate is applied to all forecast periods.

        If list, each element is interpreted as the growth rate for one forecast
        period. If the list is shorter than the required forecast horizon, the
        last value is repeated. If the list is longer than the forecast horizon,
        a ValueError is raised.

    Examples
    --------
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> import pandas as pd
    >>> y = pd.Series(
    ...     [100, 105, 110],
    ...     index=pd.period_range("2024-01", periods=3, freq="M"),
    ... )
    >>> fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    >>> forecaster = DeterministicGrowthForecaster(growth_rate=0.02)
    >>> forecaster.fit(y)
    DeterministicGrowthForecaster(growth_rate=0.02)
    >>> forecaster.predict(fh)
    2024-04    112.2000
    2024-05    114.4440
    2024-06    116.7329
    Freq: M, Name: y_pred, dtype: float64

    Notes
    -----
    For a scalar growth_rate, forecasts are computed as:

        y_hat[t+h] = y[t] * (1 + growth_rate) ** h

    For a list of growth rates, forecasts are computed as:

        y_hat[t+h] = y[t] * product((1 + growth_rate[j]) for j=1,...,h)
    """

    _tags = {
        "requires-fh-in-fit": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": False,
        "capability:insample": False,
        "handles-missing-data": False,
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
    }

    def __init__(self, growth_rate=0.02):
        self.growth_rate = growth_rate
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """
        Fit the forecaster.

        This forecaster only stores the last observed value of the target series.
        The stored value is used as the anchor for future deterministic
        projections.

        Parameters
        ----------
        y : pd.Series
            Target time series.
        X : optional, default=None
            Exogenous variables. Ignored by this forecaster.
        fh : optional, default=None
            Forecasting horizon. Not required during fitting.

        Returns
        -------
        self :
            Fitted forecaster.
        """
        self.last_value_ = y.iloc[-1]
        return self

    def _resolve_growth_rates(self, n_periods):
        """
        Convert growth_rate into an array with one rate per forecast period.

        Parameters
        ----------
        n_periods : int
            Maximum number of forecast periods required.

        Returns
        -------
        np.ndarray
            Array of growth rates with length equal to n_periods.

        Raises
        ------
        TypeError
            If growth_rate is not an int, float, or list of int/float values.
        ValueError
            If growth_rate is an empty list or longer than n_periods.
        """
        growth_rate = self.growth_rate

        if isinstance(growth_rate, bool):
            raise TypeError(
                "growth_rate must be an int, float, or a list of int/float values. "
                "Boolean values are not accepted."
            )

        if isinstance(growth_rate, numbers.Real):
            return np.repeat(float(growth_rate), n_periods)

        if isinstance(growth_rate, list):
            if len(growth_rate) == 0:
                raise ValueError("growth_rate cannot be an empty list.")

            if len(growth_rate) > n_periods:
                raise ValueError(
                    "growth_rate cannot be longer than the forecast horizon. "
                    f"Received {len(growth_rate)} rates for {n_periods} periods."
                )

            if any(isinstance(rate, bool) for rate in growth_rate):
                raise TypeError(
                    "Boolean values are not accepted inside growth_rate."
                )

            if not all(isinstance(rate, numbers.Real) for rate in growth_rate):
                raise TypeError(
                    "All elements in growth_rate must be int or float values."
                )

            growth_rates = np.asarray(growth_rate, dtype=float)

            if len(growth_rates) < n_periods:
                n_missing = n_periods - len(growth_rates)
                last_rate = growth_rates[-1]

                growth_rates = np.concatenate(
                    [
                        growth_rates,
                        np.repeat(last_rate, n_missing),
                    ]
                )

            return growth_rates

        raise TypeError(
            "growth_rate must be an int, float, or a list of int/float values."
        )

    def _predict(self, fh, X=None):
        """
        Forecast future values using cumulative percentage growth.

        Parameters
        ----------
        fh : ForecastingHorizon, list, np.ndarray, or index-like
            Forecasting horizon.
        X : optional, default=None
            Exogenous variables. Ignored by this forecaster.

        Returns
        -------
        pd.Series
            Forecasted values.
        """
        fh = self._check_fh(fh)

        steps = fh.to_relative(self.cutoff).to_numpy()

        if np.any(steps <= 0):
            raise ValueError(
                "DeterministicGrowthForecaster is designed for out-of-sample "
                "forecasting only. Use positive forecast horizons, such as "
                "[1, 2, 3]."
            )

        n_periods = int(np.max(steps))

        growth_rates = self._resolve_growth_rates(n_periods=n_periods)

        cumulative_growth = np.cumprod(1 + growth_rates)

        y_pred = self.last_value_ * cumulative_growth[steps - 1]

        index = fh.to_absolute(self.cutoff).to_pandas()

        return pd.Series(y_pred, index=index, name="y_pred")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set. Currently unused.

        Returns
        -------
        list of dict
            Parameter settings used by sktime's estimator checks.
        """
        return [
            {"growth_rate": 0.02},
            {"growth_rate": [0.01, 0.02, 0.03]},
            {"growth_rate": -0.01},
        ]
