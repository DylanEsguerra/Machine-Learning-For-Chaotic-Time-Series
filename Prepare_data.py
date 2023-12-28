import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class DatasetLoader:
    def __init__(self, dataset, lag, forecast_steps, batch_size=32, train_split=0.8):
        self.batch_size = batch_size
        self.train_split = train_split
        self.X, self.y = self.create_dataset(dataset, lag, forecast_steps)
        self.split_data()

    def create_dataset(self, dataset, lag, forecast_steps):
        X, y = [], []
        for i in range(len(dataset) - lag - forecast_steps + 1):
            feature = dataset[i:i + lag]
            target = dataset[i + forecast_steps:i + lag + forecast_steps]
            X.append(feature)
            y.append(target)

        X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to a single NumPy array
        y = torch.tensor(np.array(y), dtype=torch.float32)  # Convert to a single NumPy array

        return X, y

    def split_data(self):
        split = int(self.train_split * len(self.X))
        self.X_train, self.X_test = self.X[:split, :], self.X[split:, :]
        self.y_train, self.y_test = self.y[:split, :], self.y[split:, :]

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def get_data_loaders(self):
        train_loader = DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        return train_loader, test_loader

