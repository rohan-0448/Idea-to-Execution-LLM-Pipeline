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
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object (replace with your actual API key)
napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# Download data, commented out if data already exists
# data = napi.get_data(data_type="training")  # BytesIO
# file_path = "training_data.zip"  # Change the file name as needed
# with open(file_path, 'wb') as f:
#     f.write(data.getbuffer())
# print(f"Data downloaded and saved to {file_path}")

# Import data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe
df = df.drop('id', axis=1)  # Drop the id column
df.dropna(inplace=True)  # Drop rows with missing values in the training data

# Separate features and labels
X = df.iloc[:, :-2].values  # Keep as numpy array for scaling
y = df.iloc[:, -1].values  # Keep as numpy array for transformation

# Convert labels from [-1, 1] to [0, 4]
y = np.array([int(2 * (score_10 + 1)) for score_10 in y])


# Feature scaling using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
INPUT_SIZE = X.shape[1]  # Get input size from the data
OUTPUT_SIZE = 5  # Number of output classes
HIDDEN_SIZE = 100  # Size of the hidden layer
BATCH_SIZE = 64  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer
NUM_EPOCHS = 30  # Number of training epochs
VALIDATION_SPLIT = 0.2  # Ratio of data to use for validation


# Time Series Split Function
def time_series_split(X, y, split_ratio=0.8):
    """Splits time series data into training and validation sets."""
    split_index = int(len(X) * split_ratio)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    return X_train, X_val, y_train, y_val

# Split data into training and validation *before* converting to tensors
X_train, X_val, y_train, y_val = time_series_split(X, y, split_ratio=1 - VALIDATION_SPLIT)

# Convert to tensors *after* splitting
X_train = torch.tensor(X_train, dtype=torch.float)
X_val = torch.tensor(X_val, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)


# Define Dataset class
class NuminDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Features used for training
        self.labels = labels  # Labels used for training

    def __len__(self):
        return len(self.labels)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        sample = self.data[idx]  # Get sample at index 'idx'
        label = self.labels[idx]  # Get label at index 'idx'
        return sample, label  # Return sample and label


# Define MLP class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # Input linear layer with dimensions input_size to hidden_size
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer with dimensions hidden_size to hidden_size
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer

    def forward(self, X):
        out = self.l1(X)  # First linear layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the first linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l2(X)  # Apply second linear layer
        out = self.bn2(out)  # Batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the second linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l3(X)  # Apply third linear layer
        return out


# Instantiate dataset
train_dataset = NuminDataset(X_train, y_train)
val_dataset = NuminDataset(X_val, y_val)


# Instantiate dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate MLP model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)  # Move model to the selected device

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float('inf')  # Initialize best validation loss
model_fp = config['file_paths']['output_directory'] + '/mlp.pth'  # Path to save the model

for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):  # Iterate through the dataset for NUM_EPOCHS
    # Training phase
    mlp.train()  # Set the model to training mode
    train_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(DEVICE)  # Move features to the correct device
        labels = labels.to(DEVICE)  # Move labels to the correct device

        # Forward pass
        outputs = mlp(features)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss

        # Backward pass
        optimizer.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    mlp.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for features, labels in val_loader:
            features = features.to(DEVICE)  # Move features to the correct device
            labels = labels.to(DEVICE)  # Move labels to the correct device

            outputs = mlp(features)  # Get model predictions
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Update the total number of samples
            correct += (predicted == labels).sum().item()  # Update the number of correctly predicted samples

    avg_val_loss = val_loss / len(val_loader)
    accuracy = (correct / total) * 100  # Calculate accuracy

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(mlp.state_dict(), model_fp)  # Save the model
        print(f"Model saved to {model_fp}")  # Print message to confirm model has been saved

print("Training finished!")