# Import libraries
from numin import NuminAPI  # Library to download data and make submissions to the Numin platform
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computation library
from tqdm import tqdm  # Progress bar library
import torch  # Deep learning library
import torch.nn as nn  # Neural network library
import os  # To access files and directories
import time  # To measure time
from torch.utils.data import Dataset, DataLoader  # To create custom datasets and dataloaders
import yaml

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
napi = NuminAPI(api_key='API-KEY')

# Load data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe

# Prepare data for LSTM (sequential data)
# Store sequences and labels in a list
sequences, labels = [], []

def create_sequences(data, seq_length):
    """Convert the data into sequences of length `seq_length`."""
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length), :-2].values  # Features (sequence)
        y = data.iloc[i + seq_length, -1]  # Label (next value)
        y = int(2 * (y + 1))  # Convert labels from [-1, 1] to [0, 4]
        sequences.append(x)
        labels.append(y)

    return None

# Process data for each unique 'id'
for id in df['id'].unique():
    # Get data for each 'id'
    data = df[df['id'] == id].drop('id', axis=1)
    data.dropna(inplace=True)  # Drop rows with missing values in the training data

    SEQ_LENGTH = 10  # Define sequence length
    create_sequences(data, SEQ_LENGTH)  # Create sequences

X = np.array(sequences) 
y = np.array(labels)

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
INPUT_SIZE = X.shape[2]  # Number of features per timestep
HIDDEN_SIZE = 128  # Size of the hidden layer
OUTPUT_SIZE = 5  # Number of output classes
NUM_LAYERS = 4  # Number of LSTM layers

# Define LSTM class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Define Dataset class
class NuminDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

# Instantiate dataset and dataloader
data = NuminDataset(X, y)
dataloader = DataLoader(data, batch_size=32, shuffle=True)
    
# Instantiate LSTM model
model = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layer_dim=NUM_LAYERS, output_dim=OUTPUT_SIZE).to(DEVICE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
NUM_EPOCHS = 50

model.train()  # Set model to training mode
for epoch in tqdm(range(NUM_EPOCHS)):
    for i, (sample, labels) in enumerate(dataloader):

        sample = sample.to(DEVICE)  # Move the data to the correct device
        labels = labels.to(DEVICE)  # Move the labels to the correct device

        outputs = model(sample)  # Get model predictions
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Save model to saved_models directory
model_fp = config['file_paths']['output_directory'] + '/lstm.pth'  # Path to save the model
torch.save(model.state_dict(), model_fp)  # Save the model
print(f"Model saved to {model_fp}")  # Print message to confirm model has been saved