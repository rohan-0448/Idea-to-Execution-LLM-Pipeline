# Import libraries
from numin import NuminAPI  # Library to download data and make submissions to the Numin platform
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computation library
from tqdm import tqdm  # Progress bar library
import torch  # Deep learning library
import torch.nn as nn  # Neural network library
import os  # To access files and directories
import time  # To measure time
from torch.utils.data import Dataset, DataLoader, random_split  # To create custom datasets and dataloaders
import yaml
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
# napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')  #Commented out because I cannot use the api key.

# Load data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe

# --- Data Preprocessing ---
def preprocess_data(df, seq_length=10):
    """
    Preprocesses the data for LSTM, including feature scaling, sequence creation, and target transformation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        seq_length (int): Length of the sequence for LSTM.

    Returns:
        tuple: (X, y) where X is a NumPy array of sequences and y is a NumPy array of labels.
    """

    X, y = [], []
    for id in df['id'].unique():
        data = df[df['id'] == id].drop('id', axis=1).copy()  # Create a copy to avoid modifying the original DataFrame

        # Drop rows with missing values specific to the current 'id'
        data.dropna(inplace=True)

        if len(data) < seq_length:
             print(f"Skipping id {id} due to insufficient data after dropping NaNs.")
             continue  # Skip to the next 'id'

        # Feature scaling (fit scaler only on training data)
        numerical_features = data.columns[:-1]  # Assuming last column is the target
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Create sequences
        x_temp, y_temp = create_sequences(data, seq_length)
        X.extend(x_temp)
        y.extend(y_temp)

    return np.array(X), np.array(y)

def create_sequences(data, seq_length):
    """Convert the data into sequences of length `seq_length`."""
    x_list, y_list = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length), :-2].values  # Features (sequence)
        y = data.iloc[i + seq_length, -1]  # Label (next value)
        y = int(2 * (y + 1))  # Convert labels from [-1, 1] to [0, 4]
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list

# --- Model Definition ---
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
# Hyperparameters (consider moving these to the config file)
SEQ_LENGTH = 10
INPUT_SIZE = None  # Will be determined after preprocessing
HIDDEN_SIZE = 128  # Size of the hidden layer
OUTPUT_SIZE = 5  # Number of output classes
NUM_LAYERS = 4  # Number of LSTM layers
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class NuminDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_fp):
    best_val_loss = float('inf')  # Initialize best validation loss
    model.train()  # Set model to training mode

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        for i, (sample, labels) in enumerate(train_loader):
            sample = sample.to(device)  # Move the data to the correct device
            labels = labels.to(device)  # Move the labels to the correct device

            outputs = model(sample)  # Get model predictions
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for sample, labels in val_loader:
                sample = sample.to(device)
                labels = labels.to(device)
                outputs = model(sample)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        # Save the model if validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_fp)  # Save the model
            print(f"Best model saved to {model_fp}")  # Print message to confirm model has been saved

        model.train()  # Set model back to training mode

# --- Main Execution ---
if __name__ == "__main__":  # Protect the main execution block

    # 1. Data Preprocessing
    X, y = preprocess_data(df, SEQ_LENGTH)

    # Check if any data remains after preprocessing
    if len(X) == 0:
        raise ValueError("No data available after preprocessing. Check data and sequence length.")

    INPUT_SIZE = X.shape[2]  # Determine input size after preprocessing

    # 2. Dataset and Dataloaders
    dataset = NuminDataset(X, y)

    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Instantiate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Instantiation
    model = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layer_dim=NUM_LAYERS, output_dim=OUTPUT_SIZE).to(DEVICE)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training
    model_fp = config['file_paths']['output_directory'] + '/best_lstm.pth'
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, model_fp)

    # 6. Save Final Model
    model_fp = config['file_paths']['output_directory'] + '/lstm.pth'  # Path to save the model
    torch.save(model.state_dict(), model_fp)  # Save the model
    print(f"Final model saved to {model_fp}")  # Print message to confirm model has been saved