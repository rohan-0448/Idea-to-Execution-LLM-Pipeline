import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Constants
DEVICE = torch.device('mps')  # or 'cuda' if available
INPUT_SIZE = None  # Determined after loading data
HIDDEN_SIZE = 100
OUTPUT_SIZE = 2  # Mean and standard deviation

# Data Loading and Preprocessing
training_data_fp = './training_data/df_val_31-May-2024.csv'
try:
    df = pd.read_csv(training_data_fp)
    df = df.drop('id', axis=1)
    df.dropna(inplace=True)

    X = df.iloc[:, :-2].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    # Data scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    INPUT_SIZE = len(X[0])
    print(f"Input size: {INPUT_SIZE}") # Ensure input size is printed after data is loaded and preprocessed

except FileNotFoundError:
    print(f"Error: Training data file not found at {training_data_fp}")
    exit()
except Exception as e:
    print(f"Error loading or preprocessing data: {e}")
    exit()

# Dataset Class
class NuminDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# LSTM Model
class LSTMDistributionOutput(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMDistributionOutput, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        mean = out[:, 0]
        std = torch.exp(out[:, 1])

        return mean, std

# Instantiate model (after input size is known)
if INPUT_SIZE is not None:
    lstm_dist = LSTMDistributionOutput(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
else:
    print("Error: INPUT_SIZE is None. Model cannot be initialized.")
    exit()

# Dataset and Dataloader
dataset = NuminDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Loss Function
def negative_log_likelihood(mean, std, target):
    dist = Normal(mean, std)
    log_prob = dist.log_prob(target)
    return -log_prob.mean()

# Optimizer
optimizer = torch.optim.Adam(lstm_dist.parameters(), lr=0.001)

# Training Loop
NUM_EPOCHS = 30
lstm_dist.to(DEVICE)

for epoch in tqdm(range(NUM_EPOCHS)):
    for i, (features, labels) in enumerate(dataloader):
        inputs = features.unsqueeze(1).to(DEVICE)
        labels = labels.to(DEVICE)

        mean, std = lstm_dist(inputs)
        loss = negative_log_likelihood(mean, std, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# Save Model
model_fp = './saved_models/lstm_dist.pth'
os.makedirs(os.path.dirname(model_fp), exist_ok=True)  # Create directory if it doesn't exist
torch.save(lstm_dist.state_dict(), model_fp)
print(f"Model saved to {model_fp}")