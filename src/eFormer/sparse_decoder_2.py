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

import sys
import os

from eFormer.embeddings_2 import PositionalEncoding
from eFormer.sparse_attention_2 import DetSparseAttentionModule

# %% [markdown]
# # Sparse Decoder

# %%
class SparseDecoder(nn.Module):
    def __init__(self, d_model, n_heads, encoder_output_dim, forecast_horizon=1, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(SparseDecoder, self).__init__()
        self.d_model = encoder_output_dim[-1]
        self.batch_size = encoder_output_dim[0]
        self.n_heads = n_heads
        self.forecast_horizon = forecast_horizon
        d_ff = d_ff or 4*self.d_model
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Initialize PositionalEncoding
        self.pos_encoder = PositionalEncoding(self.batch_size, self.d_model, max_len)

        # Sparse Attention Module for cross attention
        self.cross_attention = DetSparseAttentionModule(
            d_model=self.d_model,
            n_heads=self.n_heads,
            prob_sparse_factor=5,
            seq_len=encoder_output_dim[1]
            )

        # Feed-forward network components
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=self.d_model, kernel_size=1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        # Output layer
        self.output_layer = nn.Linear(self.d_model, 1)

    def forward(self, encoder_output, attn_mask=None):
        # Generate positional encodings
        dummy_input = torch.zeros(self.batch_size, self.forecast_horizon, self.d_model)
        pos_encodings = self.pos_encoder(dummy_input)

        # Apply encoder-decoder attention using positional encodings as queries and encoder outputs as keys and values
        attn_output = self.cross_attention(encoder_output, encoder_output, encoder_output).mean(1)
        attn_output = self.norm1(attn_output + self.dropout(attn_output)).unsqueeze(1)

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
    def __init__(self, d_model, n_heads, seq_len, batch_size, forecast_horizon=1, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(DetSparseDecoder, self).__init__()
        self.SparseDecoder = SparseDecoder(
            encoder_output_dim=[batch_size, seq_len, d_model],
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, encoder_output):
        # calculate attention
        attention_output, weights = self.SparseDecoder(encoder_output)

        return attention_output.squeeze(1), weights.squeeze(1)

# %%
class ProbSparseDecoder(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, batch_size, forecast_horizon=1, max_len=5000, d_ff=None, dropout=0.1, activation="relu"):
        super(ProbSparseDecoder, self).__init__()
        self.SparseDecoder_mean = SparseDecoder(
            encoder_output_dim=[batch_size, seq_len, d_model],
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )
        self.SparseDecoder_var = SparseDecoder(
            encoder_output_dim=[batch_size, seq_len, d_model],
            forecast_horizon=1,
            d_model=d_model,
            n_heads=n_heads
        )

    def forward(self, encoder_output):
        # split tensor to extract mean and variance
        output_mean = encoder_output[0]
        output_variance = encoder_output[1]
        
        # calculate attention
        attention_output_mean, weights_mean = self.SparseDecoder_mean(output_mean)
        attention_output_var, weights_var = self.SparseDecoder_var(output_variance)

        # Combine the processed means and variances
        combined_output = torch.stack([attention_output_mean.squeeze(1), attention_output_var.squeeze(1)], dim=0)
        combined_weights = torch.stack([weights_mean.squeeze(1), weights_var.squeeze(1)], dim=0)

        return combined_output, combined_weights