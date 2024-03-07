# %%
# Standard Libraries
import pandas as pd

# Machine Learning
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# %% [markdown]
# Training Data Set
# recurrent forecast
# 
# 12 hours look back to predict next value

# %%
class TimeSeriesDataProcessor:
    def __init__(self, dataframe, forecast, look_back, batch_size=64, train_size=0.7, test_size=0.5, random_state=42):
        self.dataframe = dataframe
        self.forecast = forecast
        self.look_back = look_back
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state

    def padding_data(self, dataframe):
        remainder = dataframe.shape[0] % self.batch_size
        if remainder == 0:
            return dataframe # Already divisible by batch size
        discard = remainder
        if isinstance(dataframe, pd.DataFrame):
            return dataframe[discard:]

    def shifted_data(self):
        data = self.dataframe
        forecast = self.forecast
        look_back = self.look_back
        shifts = range(forecast, look_back + forecast)
        variables = data.columns

        shifted_columns = []
        for column in variables:
            for i in shifts:
                shifted_df = data[[column]].shift(i)
                shifted_df.rename(columns={column: f"{column} (lag {i})"}, inplace=True)
                shifted_columns.append(shifted_df)
        
        data_shifted = pd.concat([data] + shifted_columns, axis=1)
        data_shifted.dropna(inplace=True)

        return data_shifted

    def prepare_datasets(self):
        try:
            s_df = self.shifted_data().drop(['Wind speed (m/s)'], axis=1)
        except KeyError:
            s_df = self.shifted_data().copy()

        # Splitting dataset
        df_train, df_rem = train_test_split(s_df, train_size=self.train_size, random_state=self.random_state)
        df_eval, df_test = train_test_split(df_rem, test_size=self.test_size, random_state=self.random_state)

        df_train = self.padding_data(df_train)
        df_eval = self.padding_data(df_eval)
        df_test = self.padding_data(df_test)

        # Wrapping datasets
        self.train_dataset = TimeSeriesDataset(df_train)
        self.test_dataset = TimeSeriesDataset(df_test)
        self.eval_dataset = TimeSeriesDataset(df_eval)

    def create_dataloaders(self):
        self.prepare_datasets()

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)

        return self.train_loader, self.test_loader, self.eval_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe.iloc[:, 0].values
        self.features = dataframe.iloc[:, 1:].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return features, labels