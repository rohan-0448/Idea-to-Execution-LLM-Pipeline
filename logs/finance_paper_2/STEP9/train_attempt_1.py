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
from sklearn.model_selection import StratifiedKFold

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
# napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
OUTPUT_SIZE = 5  # Number of output classes
HIDDEN_SIZE = 100  # Size of the hidden layer
NUM_ASSETS = 100 # Example Value, need to replace with the number of assets in the data
EMBEDDING_DIM = 20 # Example Value, need to replace with proper embedding dimension

# Define Dataset class
class NuminDataset(Dataset):
    def __init__(self, data, stock_ids, labels):
        self.data = torch.tensor(data, dtype=torch.float)  # Numerical features
        self.stock_ids = torch.tensor(stock_ids, dtype=torch.long)  # Stock IDs
        self.labels = torch.tensor(labels, dtype=torch.long)  # Labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        numerical_sample = self.data[idx]
        stock_id = self.stock_ids[idx]
        label = self.labels[idx]
        return numerical_sample, stock_id, label

# Define MLP class
class MLP(nn.Module):
    def __init__(self, num_numerical_features, num_stocks, embedding_dim, hidden_size, output_size):
        super(MLP, self).__init__()

        self.embedding = nn.Embedding(num_stocks, embedding_dim) #Embedding layer
        self.l1 = nn.Linear(num_numerical_features + embedding_dim, hidden_size)  # Input linear layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, numerical_data, stock_ids):
        # Stock IDs are passed separately
        embedded = self.embedding(stock_ids)  # Embed the stock IDs

        #Concatenate the numerical data with the embeddings
        combined = torch.cat((numerical_data, embedded), dim=1)

        out = self.l1(combined)  # First linear layer
        out = self.relu(out)  # Apply activation function
        out = self.l2(out)  # Apply second linear layer
        out = self.relu(out)  # Apply activation function
        out = self.l3(out)  # Apply third linear layer
        return out

def create_stratified_dataloader(X, y, stock_ids, batch_size, n_splits=5): # Number of splits is the number of batches
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #shuffle data
    for train_index, val_index in skf.split(X, y):
        X_batch, y_batch, stock_ids_batch = X[train_index], y[train_index], stock_ids[train_index] #Create a batch of data
        X_batch = torch.tensor(X_batch, dtype=torch.float32)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        stock_ids_batch = torch.tensor(stock_ids_batch, dtype=torch.long)
        yield X_batch, y_batch, stock_ids_batch

def update_pvm(pvm, predictions, learning_rate=0.1): #prediction should be a single value
    #The classification should be transalted to a new portfolio weights.
    #For example, we can create a dictionary for this
    weight_update_map = {
        0: -0.02, #Represents -1
        1: -0.01, #Represents -0.5
        2: 0.00, #Represents 0
        3: 0.01, #Represents 0.5
        4: 0.02 #Represents 1
    }
    update_value = weight_update_map[predictions] #pick the update values
    pvm = pvm + update_value #update the weights based on the result

    #Normalise the vector
    pvm[pvm < 0] = 0  # Ensure no negative weights
    pvm /= pvm.sum()  # Normalize to sum to 1
    return pvm

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculates the Sharpe ratio given a series of returns."""
    excess_returns = returns - risk_free_rate / 252 #Daily risk free rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio

def calculate_mdd(portfolio_values):
    """Calculates the Maximum Drawdown given a series of portfolio values."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    mdd = np.min(drawdown)
    return mdd

def train_agent(network, optimizer, loss_function, num_epochs, X, y, stock_ids, pvm, batch_size=4, n_splits=5):
    for epoch in range(num_epochs):
        train_loss = 0.0
        for X_batch, y_batch, stock_ids_batch in create_stratified_dataloader(X, y, stock_ids, batch_size, n_splits):
            #Move data to device
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            stock_ids_batch = stock_ids_batch.to(DEVICE)

            #Forward pass
            outputs = mlp(X_batch, stock_ids_batch)
            loss = criterion(outputs, y_batch)
            train_loss += loss.item()

            #Backward pass
            optimizer.zero_grad()  # Zero out the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

# Instantiate MLP model
# Assuming num_numerical_features and num_stocks are defined based on your data
num_numerical_features = 10  # Example, replace with actual number of numerical features
num_stocks = 50 # Example, replace with actual number of stocks
mlp = MLP(num_numerical_features, num_stocks, EMBEDDING_DIM, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)  # Move model to the selected device

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001) # Reduce learning rate

# PVM Implementation (Example)
PVM = np.full(NUM_ASSETS, 1 / NUM_ASSETS)  # Initialize with uniform weights

# Training loop
NUM_EPOCHS = 30  # Number of training epochs
best_val_loss = float('inf')  # Initialize best validation loss to infinity

# Download data, commented out if data already exists
# napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')
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

# Assuming you have a column named 'stock_id'
X = df.drop(['score_10', 'next_day_direction', 'stock_id'], axis=1).values.tolist()  # Separate numerical features
stock_ids = df['stock_id'].values.tolist()  # Separate stock IDs
y = df['next_day_direction'].values.tolist()  # Store labels
y = [int(2 * (score_10 + 1)) for score_10 in y]  # Convert labels from [-1, 1] to [0, 4] to convert into a classification problem

# Split data into training and validation sets
X_train, X_val, y_train, y_val, stock_ids_train, stock_ids_val = train_test_split(
    X, y, stock_ids, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_agent(mlp, optimizer, criterion, NUM_EPOCHS, X_train, y_train, stock_ids_train, PVM)

#Evaluation
portfolio_values = []
returns = []
initial_portfolio_value = 100000 #Starting value of $100,000
portfolio_value = initial_portfolio_value
previous_portfolio_value = initial_portfolio_value
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation during validation
    for numerical_data, stock_id, labels in NuminDataset(X_val, stock_ids_val, y_val):
        numerical_data = numerical_data.to(DEVICE)  # Move features to the correct device
        stock_id = stock_id.to(DEVICE)
        labels = labels.to(DEVICE)  # Move labels to the correct device

        outputs = mlp(numerical_data.unsqueeze(0), stock_id.unsqueeze(0))  # Get model predictions, add batch dimension
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        PVM = update_pvm(PVM, predicted, 0.1)

        # Assuming you have some way to simulate trading and calculate returns based on PVM weights
        # This is a placeholder; replace with your actual trading simulation logic
        simulated_return = np.random.normal(0.001, 0.01) #Simulate some returns
        portfolio_value += portfolio_value * simulated_return

        returns.append(simulated_return)
        portfolio_values.append(portfolio_value) #Add the new portfolio values to a list

sharpe_ratio = calculate_sharpe_ratio(np.array(returns))
mdd = calculate_mdd(np.array(portfolio_values))
accuracy = 100 * correct / total

print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {mdd:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

print('Finished Training')