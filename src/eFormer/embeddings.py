# Libraries
import math

# machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Probabilistic Embeddings

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
    # print(f"tau: {tau.shape} \n w: {w.shape}")
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    
    return torch.cat([v1, v2], -1)

class ProbabilisticSineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProbabilisticSineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, self.out_features - 1))
        self.b = nn.Parameter(torch.randn(self.out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        # Calculate mean
        mean = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        
        # Calculate variance (use another set of weights and biases, ensure positive variance)
        variance = F.softplus(t2v(tau, self.f, self.out_features, 
                                  torch.randn_like(self.w), torch.randn_like(self.b), 
                                  torch.randn_like(self.w0), torch.randn_like(self.b0)))
        
        return torch.cat([mean, variance], -1)

class ProbabilisticCosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProbabilisticCosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, self.out_features - 1))
        self.b = nn.Parameter(torch.randn(self.out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        # Calculate mean
        mean = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        
        # Calculate variance (use another set of weights and biases, ensure positive variance)
        variance = F.softplus(t2v(tau, self.f, self.out_features, 
                                  torch.randn_like(self.w), torch.randn_like(self.b), 
                                  torch.randn_like(self.w0), torch.randn_like(self.b0)))
        
        return torch.cat([mean, variance], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape [batch_size, seq_len]
        seq_len = x.size(-1)
        # Expanding to match the batch size of x
        pos_encoding = self.pe[:, :seq_len]
        
        return pos_encoding


class Encoding(nn.Module):
    def __init__(self, in_features, out_features, max_len=5000):
        super(Encoding, self).__init__()
        self.out_features = out_features
        self.time2vec = ProbabilisticSineActivation(in_features, out_features) # Or CosineActivation
        self.positional_encoding = PositionalEncoding(out_features, max_len)
        
    def forward(self, tau):
        # Compute Time2Vec embeddings
        embeddings = self.time2vec(tau)

        # split embeddings
        embedding_mean, _ = torch.split(embeddings, self.out_features, dim=-1)

        # Add positional encodings
        seq_len = tau.size(1)
        pos_encodings = self.positional_encoding(embedding_mean).to(tau.device)
        
        return embedding_mean + pos_encodings[:seq_len, :]


class ProbEncoding(nn.Module):
    def __init__(self, in_features, out_features, max_len=5000):
        super(ProbEncoding, self).__init__()
        self.out_features = out_features
        # Adjustments for out_features to account for separate mean/variance in Probabilistic Activation
        self.time2vec = ProbabilisticSineActivation(in_features, out_features)
        self.positional_encoding = PositionalEncoding(out_features, max_len)
        
    def forward(self, tau):
        # Generate embeddings (mean and variance separately)
        embeddings = self.time2vec(tau)  # Expecting [1, seq_len, features] for embeddings
        
        # split embeddings
        embedding_mean, embedding_var = torch.split(embeddings, self.out_features, dim=-1)
        
        # Apply positional encodings separately to mean and variance
        pos_encoded_mean = self.positional_encoding(embedding_mean).to(tau.device)
        pos_encoded_var = self.positional_encoding(embedding_var).to(tau.device)

        # Combine mean and variance after applying positional encoding
        seq_len = tau.size(1)
        combined_embeddings = torch.stack([embedding_mean + pos_encoded_mean[:seq_len, :], embedding_var + pos_encoded_var[:seq_len, :]], dim=0)

        return combined_embeddings