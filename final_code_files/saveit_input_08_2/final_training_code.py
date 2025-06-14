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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import zipfile

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# Configuration values
training_data_fp = config['file_paths']['training_data']
data_dir = os.path.dirname(training_data_fp)  # Get the directory
zip_file_path = os.path.join(data_dir, "training_data.zip")  # Full path to zip file

# Download data if it doesn't exist
if not os.path.exists(training_data_fp):  # Check if CSV exists (unzipped)
    if not os.path.exists(zip_file_path): #Check if zip exists
        print("Downloading training data...")
        data = napi.get_data(data_type="training")  # BytesIO
        with open(zip_file_path, 'wb') as f:
            f.write(data.getbuffer())
        print(f"Data downloaded and saved to {zip_file_path}")

    #Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Data extracted to {data_dir}")
else:
    print("Training data already exists. Skipping download.")

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For GPU, if used
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe
df = df.drop('id', axis=1)  # Drop the id column
df.dropna(inplace=True)  # Drop rows with missing values in the training data

X = df.iloc[:, :-2].values.tolist()  # Separate features out from the labels
y = df.iloc[:, -1].values.tolist()  # Store labels
y = [int(2 * (score_10 + 1)) for score_10 in y]  # Convert labels from [-1, 1] to [0, 4] to convert into a classification problem

# Data Scaling
scaler = StandardScaler()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)  # Use the same scaler fitted on training data

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
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer with dimensions hidden_size to hidden_size
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer

    def forward(self, X):
        out = self.l1(X)  # First linear layer
        out = self.bn1(out)  # Apply batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the first linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l2(out)  # Apply second linear layer
        out = self.bn2(out)  # Apply batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the second linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l3(out)  # Apply third linear layer
        return out

# Instantiate MLP model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)  # Move model to the selected device

# Instantiate datasets
train_dataset = NuminDataset(X_train, y_train)
val_dataset = NuminDataset(X_val, y_val)

# Instantiate dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)  # Lower learning rate

# Training loop
NUM_EPOCHS = 30  # Number of training epochs
best_val_loss = float('inf')  # Initialize best validation loss

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

        # Backward pass
        optimizer.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

        train_loss += loss.item()

    # Validation phase
    mlp.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for features, labels in val_dataloader:
            features = features.to(DEVICE)  # Move features to the correct device
            labels = labels.to(DEVICE)  # Move labels to the correct device

            outputs = mlp(features)  # Get model predictions
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average losses and accuracy
    train_loss = train_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)
    accuracy = (correct / total) * 100

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model if validation loss is improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_fp = config['file_paths']['output_directory'] + '/best_mlp.pth'  # Path to save the best model
        torch.save(mlp.state_dict(), model_fp)  # Save the model
        print(f"Best model saved to {model_fp}")  # Print message to confirm model has been saved

# Save final model
model_fp = config['file_paths']['output_directory'] + '/final_mlp.pth'  # Path to save the model
torch.save(mlp.state_dict(), model_fp)  # Save the model
print(f"Final model saved to {model_fp}")  # Print message to confirm model has been saved

# Load the best model (after training)
best_model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
best_model.load_state_dict(torch.load(config['file_paths']['output_directory'] + '/best_mlp.pth'))
best_model.eval() #Set to evaluation mode