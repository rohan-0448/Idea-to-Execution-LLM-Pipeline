# Import libraries
from numin import NuminAPI  # Library to download data and make submissions to the Numin platform
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computation library
from tqdm import tqdm  # Progress bar library
import torch  # Deep learning library
import torch.nn as nn  # Neural network library
import os  # To access files and directories
import time  # To measure time
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler  # To create custom datasets and dataloaders
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

#Check if necessary columns are present
required_columns = ['score_10', 'next_day_direction', 'stock_id']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# Correct Column Selection
df.dropna(inplace=True)  # Drop rows with missing values in the training data
X = df.drop(['score_10', 'next_day_direction', 'stock_id'], axis=1).values.tolist()  # Drop target and ID from features
y = df['score_10'].values.tolist()  # Target variable
y = [int(2 * (score_10 + 1)) for score_10 in y]  # Convert labels from [-1, 1] to [0, 4]


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
EMBEDDING_DIM = 50

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
    def __init__(self, input_size, hidden_size, output_size, num_stock_ids, embedding_dim):
        super(MLP, self).__init__()
        self.stock_embedding = nn.Embedding(num_stock_ids, embedding_dim)  # embedding_dim is a hyperparameter
        self.num_numerical_features = input_size
        self.l1 = nn.Linear(self.num_numerical_features + embedding_dim, hidden_size)  # Input linear layer with dimensions input_size to hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer with dimensions hidden_size to hidden_size
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU()  # Activation function

    def forward(self, X, stock_ids): #Add stock_ids
        stock_embeds = self.stock_embedding(stock_ids)
        X = torch.cat((X, stock_embeds), dim=1)
        out = self.l1(X)  # First linear layer
        out = self.relu(out)  # Apply activation function to outputs of the first linear layer
        out = self.l2(out)  # Apply second linear layer
        out = self.relu(out)  # Apply activation function to outputs of the second linear layer
        out = self.l3(out)  # Apply third linear layer
        return out

# Instantiate dataset and dataloader
train_dataset = NuminDataset(X_train, y_train)
val_dataset = NuminDataset(X_val, y_val)

# Stratified Sampling
def create_weighted_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[i] for i in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

train_sampler = create_weighted_sampler(y_train) #needs to be a tensor
train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # No need to shuffle validation data

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001) # Reduce learning rate

# Initialize PVM (Portfolio Vector Memory) - Example
PVM = {}  # Dictionary to store portfolio vectors, or tensor

# Training loop
NUM_EPOCHS = 30  # Number of training epochs
best_val_loss = float('inf')  # Initialize best validation loss to infinity

# Calculate number of stocks
num_stock_ids = len(df['stock_id'].unique())
# Instantiate MLP model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_stock_ids, EMBEDDING_DIM).to(DEVICE)  # Move model to the selected device

for epoch in tqdm(range(NUM_EPOCHS)):  # Iterate through the dataset for NUM_EPOCHS
    # Training phase
    mlp.train()  # Set the model to training mode
    train_loss = 0.0
    for i, (features, labels) in enumerate(train_dataloader):

        features = features.to(DEVICE)  # Move features to the correct device
        labels = labels.to(DEVICE)  # Move labels to the correct device
        
        # Assuming 'stock_id' is a column in your dataframe after merging Numin data
        stock_ids = torch.tensor(df['stock_id'].astype('category').cat.codes.values).to(DEVICE)  # Convert stock IDs to numerical indices

        # Forward pass
        outputs = mlp(features, stock_ids)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

        # Update PVM based on the prediction
        # Assuming 'outputs' contains the predicted portfolio weights
        # Update PVM using some logic (e.g., exponential moving average)
        #update_portfolio_vector_memory(outputs) # Placeholder to update the portfolio vector memory
        pass

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

            stock_ids = torch.tensor(df['stock_id'].astype('category').cat.codes.values).to(DEVICE)

            outputs = mlp(features, stock_ids)  # Get model predictions
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
        model_fp = config['file_paths']['output_directory'] + '/best_mlp.pth'  # Path to save the model
        torch.save(mlp.state_dict(), model_fp)  # Save the model
        print(f"Best model saved to {model_fp}")  # Print message to confirm model has been saved

print('Finished Training')
```"