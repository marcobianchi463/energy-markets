import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple

# --- Metrics here ---

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Returns mean average error"""
    return np.mean(np.abs(y_pred - y_true)).astype(float)

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Returns root mean square error"""
    return np.sqrt(np.mean((y_pred - y_true).astype('float64') ** 2))

def mape(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Returns mean average percentual error. Uses epsilon = 1.0 EUR/MWh to avoid division by zero"""
    return np.mean(np.abs(y_pred - y_true) / np.maximum(y_pred, epsilon)).astype(float) * 100.0

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": round(mae(y_pred, y_true), 3),
        "RMSE": round(rmse(y_pred, y_true), 3),
        "MAPE": round(mape(y_pred, y_true), 3),
    }

def walk_forward_cv(
        series: pd.Series,
        model_fn: Callable,
        test_size: int = 24,
        min_train_size: int = 24 * 90,  # at least 90 days of training window
        step: int = 24 * 7,             # move week by week
        **kwargs,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Walk-forward Cross Validation for time series.
    
    Args:
        series: complete timeseries with Timestamp index
        model_fn: function accepting (train: pd.Series, test_size: int) and returns a prediction array of size test_size
        test_size: number of hours to predict at each fold
        min_train_size: minimum training window size in hours
        step: number of hours between each fold

    Returns:
        metrics_df: metrics for each fold
        all_preds: predicted timeseries
        all_true: actual timeseries associated with all_preds
    """
    results = []
    all_preds = []
    all_true = []

    indices = range(min_train_size, len(series) - test_size + 1, step)

    lamb = kwargs['lamb']if 'lamb' in kwargs else None

    for i, t in enumerate(indices):
        model = None
        preds = None
        progress_string = '[' + int(20*(i+1)/len(indices))*'>' + int(20*(1-i)/len(indices))*'.' + ']'
        print(f"Progress:{(i+1)/len(indices):.0%}\t{progress_string}\033[0K\r", end='')
        train = series.iloc[:t]
        test = series.iloc[t : t + test_size]

        if len(test) < test_size:
            break

        preds, model = model_fn(train, test_size, lamb=lamb)
        metrics = compute_metrics(test.to_numpy(), preds)
        metrics["fold_start"] = series.index[t]
        results.append(metrics)

        all_preds.append(pd.Series(preds, index=test.index))
        all_true.append(test)

    metrics_df = pd.DataFrame(results).set_index("fold_start")
    all_preds = pd.concat(all_preds)
    all_true = pd.concat(all_true)

    return metrics_df, all_preds, all_true, model