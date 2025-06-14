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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
# napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

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

X = df.iloc[:, :-2].values.tolist()  # Separate features out from the labels
y = df.iloc[:, -1].values.tolist()  # Store labels
y = [int(2 * (score_10 + 1)) for score_10 in y]  # Convert labels from [-1, 1] to [0, 4] to convert into a classification problem

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
INPUT_SIZE = len(X[0])  # Get input size from the data
print(INPUT_SIZE)
OUTPUT_SIZE = 5  # Number of output classes
HIDDEN_SIZE = 100  # Size of the hidden layer

# Define Dataset class
class NuminDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)  # Features used for training
        self.labels = torch.tensor(labels, dtype=torch.long)  # Labels used for training

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
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer with dimensions hidden_size to hidden_size
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU()  # Activation function

    def forward(self, X):
        out = self.l1(X)  # First linear layer
        out = self.relu(out)  # Apply activation function to outputs of the first linear layer
        out = self.l2(out)  # Apply second linear layer
        out = self.relu(out)  # Apply activation function to outputs of the second linear layer
        out = self.l3(out)  # Apply third linear layer
        return out

# Instantiate MLP model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)  # Move model to the selected device

# Instantiate dataset and dataloader
train_dataset = NuminDataset(X_train, y_train)
val_dataset = NuminDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Shuffle training data
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # No need to shuffle validation data

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001) # Reduce learning rate

# Training loop
NUM_EPOCHS = 30  # Number of training epochs
best_val_loss = float('inf')  # Initialize best validation loss to infinity

for epoch in tqdm(range(NUM_EPOCHS)):  # Iterate through the dataset for NUM_EPOCHS
    # Training phase
    mlp.train()  # Set the model to training mode
    train_loss = 0.0
    for i, (features, labels) in enumerate(train_dataloader):

        features = features.to(DEVICE)  # Move features to the correct device
        labels = labels.to(DEVICE)  # Move labels to the correct device

        # Forward pass
        outputs = mlp(features)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

    avg_train_loss = train_loss / len(train_dataloader)

    # Validation phase
    mlp.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for features, labels in val_dataloader:
            features = features.to(DEVICE)  # Move features to the correct device
            labels = labels.to(DEVICE)  # Move labels to the correct device

            outputs = mlp(features)  # Get model predictions
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model if validation loss has decreased
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_fp = os.path.join(config['file_paths']['output_directory'], 'best_mlp.pth')  # Path to save the model
        torch.save(mlp.state_dict(), model_fp)  # Save the model
        print(f"Best model saved to {model_fp}")  # Print message to confirm model has been saved

print('Finished Training')