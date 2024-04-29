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

def reading_Windturbines(turbine_directory):

    def safe_datetime_conversion(s):
        try:
            return pd.to_datetime(s)
        except:
            return pd.NaT

    def days_since_last_maintenance(row_date, maintenance_dates):
        # Exclude None values from the maintenance_dates list before making comparisons
        preceding_maintenance_dates = [date for date in maintenance_dates if date is not None and date <= row_date]
        if not preceding_maintenance_dates:
            return float('NaN')
        last_maintenance_date = max(preceding_maintenance_dates)
        delta = (row_date - last_maintenance_date).days
        return delta

    # Columns to keep
    columns_turbine = [
        '# Date and time',
        'Wind speed (m/s)',
        'Long Term Wind (m/s)',
        'Power (kW)'
    ]
    columns_status = [
        'Timestamp end',
        'IEC category'
    ]

    # Directory containing CSV files
    directory = f'../data/Windturbinen/{turbine_directory}/'

    # Dictionary to hold DataFrames for each turbine
    turbine_dataframes = defaultdict(list)
    status_lists = defaultdict(list)

    # Get a list of CSV files in the directory
    turbine_files = [f for f in os.listdir(directory) if f.startswith(f"Turbine_Data_{turbine_directory}_") and f.endswith(".csv")]
    status_files = [f for f in os.listdir(directory) if f.startswith(f"Status_{turbine_directory}_") and f.endswith(".csv")]

    # Iterate through the status files
    for filename in tqdm(status_files, desc='Processing status files'):
        turbine_number = filename.split("_")[2]
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, skiprows=9, usecols=columns_status)
        df['Timestamp end'] = df['Timestamp end'].apply(safe_datetime_conversion)
        maintenance_dates = df[df['IEC category'] == 'Scheduled Maintenance']['Timestamp end'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isna(x) else None).unique()
        status_lists[turbine_number].extend(maintenance_dates)

    # Iterate through the turbine files
    for filename in tqdm(turbine_files, desc='Processing turbine files'):
        turbine_number = filename.split("_")[3]
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath, skiprows=9, usecols=columns_turbine)
        df['# Date and time'] = pd.to_datetime(df['# Date and time'])
        maintenance_dates = [pd.to_datetime(date) for date in status_lists[turbine_number]]
        df['Days Since Maintenance'] = df['# Date and time'].apply(lambda row_date: days_since_last_maintenance(row_date, maintenance_dates))
        turbine_dataframes[turbine_number].append(df)

    # Concatenate the DataFrames for each turbine
    for turbine_number, dfs in turbine_dataframes.items():
        turbine_dataframes[turbine_number] = pd.concat(dfs)
        turbine_dataframes[turbine_number].sort_values('# Date and time', inplace=True)
        turbine_dataframes[turbine_number] = turbine_dataframes[turbine_number].reset_index(drop=True)
        turbine_dataframes[turbine_number].set_index(pd.to_datetime(turbine_dataframes[turbine_number]['# Date and time']), inplace=True)

    print("\n dictionary keys:")
    print(turbine_dataframes.keys())
    print('\n shape for exemplary key:')
    print(turbine_dataframes[list(turbine_dataframes.keys())[0]].shape)

    return turbine_dataframes

Wind_df = reading_Windturbines('Kelmarsh')

processor = TimeSeriesDataProcessor(
    dataframe=Wind_df,
    forecast=hyperparameters['pred_len'],
    look_back=hyperparameters['seq_len'],
    batch_size=hyperparameters['batch_size'])
    
train_loader, test_loader, val_loader = processor.create_dataloaders()