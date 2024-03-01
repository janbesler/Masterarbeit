# Libraries
import pandas as pd
import numpy as np

# CRPS (continouos ranked probability score)
def crps(forecast, observations, weights):
    """
    Args:
    forecast (pd.DataFrame or np.ndarray): Forecasts from the model (ensemble).
    observations (pd.Series or np.ndarray): Observed values.
    weights (np.array): Corresponding weights for the CRPS scores, derived from sparse attention.

    Returns:
    float: Weighted mean of the CRPS for all forecasts.
    """
    # Convert to NumPy arrays if input is Pandas
    if isinstance(forecast, pd.DataFrame):
        forecast = forecast.to_numpy()
    if isinstance(observations, pd.Series):
        observations = observations.to_numpy()
    
    # Sort forecast samples
    forecast.sort(axis=0)

    # Ensure observations are broadcastable over the forecast_samples
    observations = observations[np.newaxis, :]

    # Calculate CRPS
    cumsum_forecast = np.cumsum(forecast, axis=0) / forecast.shape[0]
    crps = np.mean((cumsum_forecast - (forecast > observations).astype(float)) ** 2, axis=0)
    
    # weighted median of CRPS
    if len(crps) != len(weights):
        raise ValueError("Length of CRPS series and weights must be equal")

    weighted_sum = np.sum(crps * weights)
    total_weights = np.sum(weights)

    if total_weights == 0:
        raise ValueError("Total weight cannot be zero")

    weighted_crps = weighted_sum / total_weights
    
    return weighted_crps