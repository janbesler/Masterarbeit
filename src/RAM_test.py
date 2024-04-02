# standard
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from math import sqrt
import time

# reading data
import os
import json
from collections import defaultdict

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft, fftn, ifftn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# visuals
import matplotlib.pyplot as plt
import seaborn as sns

# measuring ressources
import time
import psutil
import GPUtil
import threading
from memory_profiler import profile

# eFormer
from eFormer.embeddings import Encoding, ProbEncoding, PositionalEncoding
from eFormer.sparse_attention import ProbSparseAttentionModule, DetSparseAttentionModule
from eFormer.loss_function import CRPS, weighted_CRPS
from eFormer.sparse_decoder import DetSparseDecoder, ProbSparseDecoder
from eFormer.Dataloader import TimeSeriesDataProcessor


# set global parameters
hyperparameters = {
    'n_heads': 4,
    'ProbabilisticModel': True,
    # embeddings
    'len_embedding': 64,
    'batch_size': 512,
    # general
    'pred_len': 1,
    'seq_len': 72,
    'patience': 7,
    'dropout': 0.05,
    'learning_rate': 6e-4,
    'WeightDecay': 1e-1,
    'train_epochs': 2,
    'num_workers': 10,
    'step_forecast': 6,
    # benchmarks
    'factor': 1,
    'output_attention': True,
    'd_model': 64,
    'c_out': 6,
    'e_layers': 2,
    'd_layers': 2,
    'activation': 'relu',
    'd_ff': 1,
    'distil': True,
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Wind_df = pd.read_csv('../data/Windturbinen/Wind_df.csv')
Wind_df = Wind_df.drop(['Unnamed: 0', 'date', 'Long Term Wind (m/s)'], axis = 1)
Wind_df = Wind_df.set_index('# Date and time')
Wind_df.index.names = [None]


processor = TimeSeriesDataProcessor(
    dataframe=Wind_df,
    forecast=hyperparameters['pred_len'],
    look_back=hyperparameters['seq_len'],
    batch_size=hyperparameters['batch_size'])
    
train_loader, test_loader, val_loader = processor.create_dataloaders()