# %% [markdown]
# # Libraries

# %%
import math

# machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% [markdown]
# # Embeddings

# %%
def t2v(
    tau, # input tensor
    f, # activation function (sin or cosin)
    out_features, # size of output vector
    w, # weights
    b, # biases
    w0, # weights for linear part of time2vec layer
    b0, # biases for linear part of time2vec layer
    arg=None # optional arguments
    ):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0

    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        # create var
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]

# %%
class Encoding(nn.Module):
    def __init__(self, in_features, out_features, max_len=5000):
        super(Encoding, self).__init__()
        self.time2vec = SineActivation(in_features, out_features) # Or CosineActivation
        self.positional_encoding = PositionalEncoding(out_features, max_len)
        
    def forward(self, tau):
        # Compute Time2Vec embeddings
        time_embeddings = self.time2vec(tau)
        # Add positional encodings
        seq_len = tau.size(1)  # Assuming (batch, seq_len, features) for tau
        pos_encodings = self.positional_encoding(time_embeddings).to(tau.device)
        return time_embeddings + pos_encodings[:seq_len, :]