# Import libraries
from numin import NuminAPI  # Library to download data and make submissions to the Numin platform
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computation library
from tqdm import tqdm  # Progress bar library
import torch  # Deep learning library
import torch.nn as nn  # Neural network library
import torch.nn.functional as F # Neural network functional library
import os  # To access files and directories
import time  # To measure time
from torch.utils.data import Dataset, DataLoader, random_split  # To create custom datasets and dataloaders
import yaml
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check if 'api_key' exists in the config
if 'api_key' not in config:
    raise ValueError("API key not found in config.yaml")

api_key = config['api_key']

# Initialize Numin API object
napi = NuminAPI(api_key=api_key)

# Download data, commented out if data already exists
# data = napi.get_data(data_type="training")  # BytesIO
# file_path = "training_data.zip"  # Change the file name as needed
# with open(file_path, 'wb') as f:
#     f.write(data.getbuffer())
# print(f"Data downloaded and saved to {file_path}")

# Import data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe
df['id'] = df['id'].astype(str) # Ensure 'id' is a string
df.dropna(inplace=True)  # Drop rows with missing values in the training data

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
NUM_ASSETS = df['id'].nunique() #Number of unique assets
HIDDEN_SIZE = 64  # Size of hidden layers
OUTPUT_SIZE = NUM_ASSETS  # Output portfolio weights for each asset
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2  # 80-20 split

# Preselected set of IDs to use (EIIE)
list_of_ids = df['id'].unique().tolist()

# Data Preprocessing
def preprocess_data(df, list_of_ids):
    # Filter IDs
    df = df[df['id'].isin(list_of_ids)]
    
    # Target variable transformation
    df['target'] = [int(2 * (score_10 + 1)) for score_10 in df['target']]
    
    # Feature Scaling (normalize historical prices to latest closing price)
    def normalize_features(group):
        # Normalize other columns relative to 'last'
        for col in group.columns:
            if col not in ['id', 'date', 'target']:
                group[col] = group[col] / group['last']
        return group
    
    df = df.groupby('id').apply(normalize_features)
    df = df.drop(['date'], axis=1, errors='ignore')

    # One-hot encode 'id'
    df = pd.get_dummies(df, columns=['id'])

    return df

df = preprocess_data(df.copy(), list_of_ids) # Preprocess the data

# Separate features and target
X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

# Define Dataset class
class PortfolioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = PortfolioDataset(X, y)

# Split dataset into training and validation sets
val_size = int(VALIDATION_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Actor-Critic Network with EIIE and CNN
class ActorCritic(nn.Module):
    def __init__(self, num_assets, hidden_size):
        super(ActorCritic, self).__init__()
        self.num_assets = num_assets
        self.hidden_size = hidden_size

        # EIIE - CNN for each asset (simplified for demonstration)
        self.cnn1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Assuming input is time series data
        self.cnn2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)

        # Example feature size after CNN layers (adjust based on your input)
        example_input_size = len(X[0])  # Replace with actual input size
        example_output = self.cnn_forward(torch.randn(1, 1, example_input_size))
        cnn_output_size = example_output.view(1, -1).size(1)

        # Actor Network - Portfolio Weights
        self.actor_fc1 = nn.Linear(cnn_output_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, num_assets)

        # Critic Network - Value Function
        self.critic_fc1 = nn.Linear(cnn_output_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, 1)

    def cnn_forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pool(x)
        x = F.relu(self.cnn2(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        # Reshape input for CNN (batch_size, num_channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension

        # CNN Feature Extraction
        x = self.cnn_forward(x)

        # Flatten CNN output
        x = x.view(x.size(0), -1)

        # Actor Network
        actor_x = F.relu(self.actor_fc1(x))
        portfolio_weights = F.softmax(self.actor_fc2(actor_x), dim=1)  # Softmax for portfolio weights

        # Critic Network
        critic_x = F.relu(self.critic_fc1(x))
        value = self.critic_fc2(critic_x)  # Value function

        return portfolio_weights, value

# Initialize the ActorCritic model
model = ActorCritic(NUM_ASSETS, HIDDEN_SIZE).to(DEVICE)

# Loss function and optimizer
actor_criterion = nn.CrossEntropyLoss()
critic_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) #Add L2 Regularization

# Training loop
best_val_loss = float('inf')  # Initialize with a very high value
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        portfolio_weights, value = model(features)
        actor_loss = actor_criterion(portfolio_weights, labels)  # Use CrossEntropyLoss for portfolio weights

        # The value function is a scalar, and labels are also scalars. So calculating MSE loss
        critic_loss = critic_criterion(value.squeeze(), labels.float())

        # Combine the losses
        loss = actor_loss + critic_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation after each epoch
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for features, labels in val_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            portfolio_weights, value = model(features)
            actor_loss = actor_criterion(portfolio_weights, labels)  # Use CrossEntropyLoss

            # The value function is a scalar, and labels are also scalars. So calculating MSE loss
            critic_loss = critic_criterion(value.squeeze(), labels.float())
            loss = actor_loss + critic_loss
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(portfolio_weights.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average losses and accuracy
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_fp = config['file_paths']['output_directory'] + '/best_model.pth'  # Path to save the model
        torch.save(model.state_dict(), model_fp)
        print(f"Best model saved to {model_fp}")

print("Training complete!")