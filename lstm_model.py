# lstm_model.py
# LSTM model used to evaluate time series, number of layers and nodes fixed 
# to make comparisons on different dynamic systems, data lengths and forecast horizons 

import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.linear(x)
        return x
