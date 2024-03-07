# %% [markdown]
# # Libraries
# 

# %%
# standard
import pandas as pd
import numpy as np
import math
from math import sqrt

# machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F

from eFormer.embeddings import PositionalEncoding
from eFormer.sparse_attention import DetSparseAttentionModule

# %% [markdown]
# # Sparse Decoder

# %%
class SparseDecoder(nn.Module):
    def __init__(self, d_model, n_heads, encoder_output_dim, forecast_horizon=22, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(SparseDecoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.forecast_horizon = forecast_horizon
        d_ff = d_ff or 4*d_model
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Initialize PositionalEncoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Sparse Attention Module for cross attention
        self.cross_attention = DetSparseAttentionModule(d_model, n_heads, prob_sparse_factor=5)

        # Feed-forward network components
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Output layer
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, encoder_output, attn_mask=None):
        # Generate positional encodings
        dummy_input = torch.zeros(self.forecast_horizon, self.d_model).unsqueeze(0)
        pos_encodings = self.pos_encoder(dummy_input)

        # Apply cross attention using positional encodings as queries and encoder outputs as keys and values
        attn_output = self.cross_attention(pos_encodings, encoder_output, encoder_output, attn_mask)
        attn_output = self.norm1(attn_output + self.dropout(attn_output))

        # Feed-forward network
        ff_output = attn_output.transpose(-1, 1)  # Prepare for conv1d
        ff_output = self.dropout(self.activation(self.conv1(ff_output)))
        ff_output = self.dropout(self.conv2(ff_output))
        ff_output = ff_output.transpose(-1, 1)  # Back to original dims
        ff_output = self.norm2(attn_output + self.dropout(ff_output))

        # Generate forecasts based on the attention output
        forecasts = self.output_layer(ff_output).squeeze(-1)
        
        return forecasts, ff_output

# %%
class DetSparseDecoder(nn.Module):
    def __init__(self, d_model, n_heads, encoder_output_dim, forecast_horizon=1, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(DetSparseDecoder, self).__init__()
        self.SparseDecoder = SparseDecoder(
            encoder_output_dim=encoder_output_dim,
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, encoder_output):
        # calculate attention
        attention_output, crps_weights = self.SparseDecoder(encoder_output)

        return attention_output, crps_weights

# %%
class ProbSparseDecoder(nn.Module):
    def __init__(self, d_model, n_heads, encoder_output_dim, forecast_horizon=1, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(ProbSparseDecoder, self).__init__()
        self.SparseDecoder_mean = SparseDecoder(
            encoder_output_dim=encoder_output_dim,
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )
        self.SparseDecoder_var = SparseDecoder(
            encoder_output_dim=encoder_output_dim,
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, encoder_output):
        # split tensor to extract mean and variance
        output_mean = encoder_output[0]
        output_variance = encoder_output[1]
        
        # calculate attention
        attention_output_mean, crps_weights_mean = self.SparseDecoder_mean(output_mean)
        attention_output_var, crps_weights_var = self.SparseDecoder_var(output_variance)

        # Combine the processed means and variances
        combined_output = torch.stack([attention_output_mean, attention_output_var], dim=0)
        combined_crps_weights = torch.stack([crps_weights_mean, crps_weights_var], dim=0)

        return combined_output, combined_crps_weights