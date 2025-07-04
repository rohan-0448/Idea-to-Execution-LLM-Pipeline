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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object - Replace with your actual API key or method for loading data
# napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')  # Replace with your API key

# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
SEQ_LENGTH = 10  # Define sequence length
HIDDEN_SIZE = 128  # Size of the hidden layer
NUM_LAYERS = 4  # Number of LSTM layers
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define LSTM class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
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
    

def acquire_training_data(file_path):
    """Acquire training data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None


def preprocess_training_data(df, feature_columns, target_column="score_10", seq_length=SEQ_LENGTH):
    """Preprocess the training data: handles missing values, scaling, and sequence creation."""
    
    # Data Cleaning
    df.dropna(inplace=True)  # Drop rows with any missing values

    # Separate features and target
    X = df[feature_columns].values
    y = df[target_column].values

    # Transform target variable from [-1, 1] to [0, 4]
    y = [int(2 * (score + 1)) for score in y]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare sequences and labels
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:(i + seq_length)])
        labels.append(y[i + seq_length])  # next value as label

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels, scaler

def create_datasets(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Splits data into training and validation datasets."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    train_dataset = NuminDataset(X_train, y_train)
    val_dataset = NuminDataset(X_val, y_val)
    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Trains the LSTM model, tracks metrics, and saves the best model."""
    best_val_loss = float('inf')  # Initialize with a very high value
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for i, (samples, labels) in enumerate(train_loader):
            samples = samples.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(samples)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient calculation during validation
            for samples, labels in val_loader:
                samples = samples.to(device)
                labels = labels.to(device)

                outputs = model(samples)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_fp = config['file_paths']['output_directory'] + '/best_lstm.pth'  # Path to save the model
            torch.save(model.state_dict(), model_fp)  # Save the model
            print(f"Best model saved to {model_fp}")

    print("Finished Training")

def main():
    """Main function to execute the training pipeline."""
    # Configuration
    training_data_fp = config['file_paths']['training_data']
    output_directory = config['file_paths']['output_directory']
    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists

    # ---------------------- ADDED CODE ----------------------
    # Load a small portion of the CSV to inspect column names
    try:
        temp_df = pd.read_csv(training_data_fp, nrows=5)  # Load first 5 rows
        print("Columns in the training data:", temp_df.columns.tolist())  # Print column names
    except FileNotFoundError:
        print(f"Error: The file {training_data_fp} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while inspecting the data: {e}")
        return
    # ---------------------- END OF ADDED CODE ----------------------

    # Define the feature columns - adjust based on your data
    # **IMPORTANT: Replace these with the ACTUAL column names from your data**
    # **Based on inspection using the added code above**

    # Assuming these columns are actually present in the data.
    # Replace with whatever is ACTUALLY in the csv.
    feature_columns = ['actual_feature1', 'actual_feature2', 'actual_feature3', 'actual_feature4', 'actual_feature5',
                       'actual_feature6', 'actual_feature7', 'actual_feature8', 'actual_feature9', 'actual_feature10']

    # 1. Data Acquisition
    df = acquire_training_data(training_data_fp)
    if df is None:
        return  # Exit if data loading fails

    # 2. Data Preprocessing
    X, y, scaler = preprocess_training_data(df.copy(), feature_columns=feature_columns)  # Pass a copy to avoid modifying the original DataFrame
    
    # Save the scaler
    torch.save(scaler, os.path.join(output_directory, 'scaler.pth'))

    # 3. Create Datasets
    train_dataset, val_dataset = create_datasets(X, y)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    INPUT_SIZE = X.shape[2] if len(X.shape) == 3 else X.shape[1]  # Number of features per timestep

    # 5. Model Instantiation
    model = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layer_dim=NUM_LAYERS, output_dim=len(np.unique(y))).to(DEVICE)

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Train the Model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

if __name__ == "__main__":
    main()