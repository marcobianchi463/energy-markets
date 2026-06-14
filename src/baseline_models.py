import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
import warnings
from typing import Callable, Dict, List, Tuple

def naive_seasonal(train: pd.Series, test_size: int, **kwargs) -> Tuple[np.ndarray, None]:
    """
    Naive 1 - Seasonal naive approach.
    Prediction = same as last week.
    No parameter, no fitting.
    """
    return train.iloc[-169 : -169 + test_size].to_numpy(), None

def naive_daily(train: pd.Series, test_size: int, **kwargs) -> Tuple[np.ndarray, None]:
    """
    Naive 2 - Daily naive approach.
    Prediction = same as yesterday.
    Simpler than seasonal, useful for comparison
    """
    return train.iloc[-25 : - 25 + test_size].to_numpy(), None

def exp_seasonal(prices: pd.Series, test_size: int, **kwargs) -> Tuple[np.ndarray, None]:
    """
    Baseline 1 - Exponential seasonal average.
    Prediction = Exponentially weighted day-of-week average.
    """
    preds = []
    lamb = kwargs['lamb'] if 'lamb' in kwargs else 1.5
    if test_size > 168: print("Warning: Test size greater than a week")
    for i in range(test_size):
        train_set = prices[-1-test_size+i::-168]
        train_size = train_set.size
        weights = np.exp(-np.arange(train_size)*np.log(lamb))
        weights /= weights.sum()
        prediction = (weights * train_set).sum()
        preds.append(prediction)
    return np.array(preds), None

def sarima_forecast(train: pd.Series, test_size: int,
                    order: tuple = (2, 0, 1),
                    seasonal_order: tuple = (1, 1, 0, 24), **kwargs) -> Tuple[np.ndarray, SARIMAXResultsWrapper]:
    """
    Baseline 3 - SARIMA with daily seasonality.
    order (p, d, q): non-seasonal component
    seasonal_order (P, D, Q, s): seasonal component, s=24 hours
    
    Note: d=0 because non-stationarity is handled
    by seasonal differentiation (D=1)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, low_memory=True)
    return fit.forecast(steps=test_size).to_numpy(), fit