# Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CRPS (continouos ranked probability score)

class CRPS(nn.Module):
    def __init__(self, custom_parameters=None):
        super(CRPS, self).__init__()
        self.custom_parameters = custom_parameters

    def forward(self, forecast, observations, weights):
        """
        Args:
        forecast (torch.Tensor): Forecasts from the model (ensemble) with shape [1, seq_len].
        observations (torch.Tensor): Observed values with shape [seq_len].
        weights (torch.Tensor): Corresponding weights for the CRPS scores, derived from sparse attention, with shape [1, seq_len, seq_len].

        Returns:
        float: Weighted mean of the CRPS for all forecasts.
        """
        forecast = forecast.squeeze(0)  # Adjusting forecast shape: [64]
        weights = weights.mean(dim=-1).squeeze(0)  # Assuming averaging is the method to obtain weights: [64]
    
        # Sorting the forecasts
        sorted_forecast, _ = torch.sort(forecast, dim=0)
        observations = observations.unsqueeze(0)  # [1, 64] for broadcasting

        # Cumulative sum of sorted forecasts
        cumsum_forecast = torch.cumsum(sorted_forecast, dim=0) / forecast.size(0)

        # Calculating CRPS
        indicator = (sorted_forecast > observations).float()
        differences = (cumsum_forecast - indicator) ** 2
        weighted_differences = differences * weights  # Apply weights to the differences
        crps = weighted_differences.mean()  # Taking mean across all weighted differences

        return crps  # Returning as a Tensor for the backward pass