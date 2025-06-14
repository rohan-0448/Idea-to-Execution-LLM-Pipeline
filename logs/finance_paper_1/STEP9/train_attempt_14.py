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
from sklearn.preprocessing import MinMaxScaler

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
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

# Assuming 'id' is stock identifier and last two columns are features
list_of_ids = df['id'].unique().tolist()[:5] # Limiting to first 5 for demonstration
df = df[df['id'].isin(list_of_ids)]
df.dropna(inplace=True)  # Drop rows with missing values in the training data

# Separate features and target
def preprocess(df, list_of_ids, config):
    # Check if the target column exists
    if 'score_10' not in df.columns:
        raise ValueError("The 'score_10' column is missing in the input CSV file.")

    # Convert labels from [-1, 1] to [0, 4]
    df['target'] = [int(2 * (score_10 + 1)) for score_10 in df['score_10']]

    # Normalize features (example: relative to latest closing price)
    close_price_column = config['columns']['close_price']  # Use config for column name

    for id in list_of_ids:
        id_df = df[df['id'] == id].copy()  # Create a copy to avoid SettingWithCopyWarning
        close_prices = id_df[close_price_column] # Assuming you have a close price column
        if not close_prices.empty: #Prevent from dividing by zero and also nan propagation in case where there is no available data
                latest_close = close_prices.iloc[-1] # Normalise by the latest closing price
        else:
                latest_close = 1.0 # If the data is not available, just set to 1

        if latest_close == 0:
                latest_close = 1.0 # Protect against zero division

        for col in id_df.columns:
            if col not in ['id', 'target', 'score_10', close_price_column]:  # Don't normalize id, target, score_10 or close
                df.loc[df['id'] == id, col] = id_df[col] / latest_close

    return df

df = preprocess(df, list_of_ids, config)


# Define constants
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use MPS if available, otherwise fall back to CPU
NUM_ASSETS = len(list_of_ids)
INPUT_SIZE = len(df.columns) - 2  # Number of features
OUTPUT_SIZE = 5  # Number of output classes
HIDDEN_SIZE = 64
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


# Define Dataset class
class NuminDataset(Dataset):
    def __init__(self, df, list_of_ids):
        self.data = []
        self.labels = []
        self.list_of_ids = list_of_ids
        self.df = df

        for id in list_of_ids:
            id_df = df[df['id'] == id].drop('id', axis = 1)

            features = id_df.iloc[:, :-1].values.tolist()
            labels = id_df.iloc[:, -1].values.tolist()

            self.data.extend(features)
            self.labels.extend(labels)

        self.data = torch.tensor(self.data, dtype=torch.float)  # Features used for training
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Labels used for training


    def __len__(self):
        return len(self.labels)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        sample = self.data[idx]  # Get sample at index 'idx'
        label = self.labels[idx]  # Get label at index 'idx'
        return sample, label  # Return sample and label


# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_assets):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.fc = nn.Linear(hidden_size, num_assets)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # CNN needs input in (batch_size, channels, sequence_length) format
        cnn_in = lstm_out.transpose(1, 2)
        cnn_out = self.cnn(cnn_in)
        # cnn_out shape: (batch_size, hidden_size, sequence_length)
        # Take the last time step's output
        last_output = cnn_out[:, :, -1]
        # last_output shape: (batch_size, hidden_size)
        portfolio_weights = self.softmax(self.fc(last_output))
        # portfolio_weights shape: (batch_size, num_assets)
        return portfolio_weights

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        value = self.fc(lstm_out[:, -1, :])  # Take the last time step
        return value


# Instantiate dataset
dataset = NuminDataset(df, list_of_ids)

# Split dataset into training and validation sets
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Instantiate dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize Actor-Critic networks
actor = Actor(INPUT_SIZE - 1, HIDDEN_SIZE, NUM_ASSETS).to(DEVICE) #Input size of features minus target feature
critic = Critic(INPUT_SIZE - 1, HIDDEN_SIZE).to(DEVICE)

# Define optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# Loss function (Mean Squared Error for Critic)
critic_criterion = nn.MSELoss()


# Training loop
best_val_loss = float('inf')
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    actor.train()
    critic.train()
    train_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(DEVICE).unsqueeze(1) # Adding sequence dimension
        labels = labels.to(DEVICE)

        # Critic update
        critic_optimizer.zero_grad()
        values = critic(features)
        advantage = labels.unsqueeze(1).float() - values  # Using labels as rewards for simplicity
        critic_loss = critic_criterion(values, labels.unsqueeze(1).float())
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update
        actor_optimizer.zero_grad()
        portfolio_weights = actor(features)
        # Reward is the return, which we approximate with -advantage (negative advantage)
        actor_loss = -(portfolio_weights * advantage.detach()).mean() #Simplified reward function
        actor_loss.backward()
        actor_optimizer.step()

        train_loss += critic_loss.item()

    # Validation loop
    actor.eval()
    critic.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE).unsqueeze(1)
            labels = labels.to(DEVICE)
            values = critic(features)
            loss = critic_criterion(values, labels.unsqueeze(1).float())
            val_loss += loss.item()


    # Print epoch statistics
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_fp = config['file_paths']['output_directory'] + '/actor.pth'
        torch.save(actor.state_dict(), model_fp)
        print(f"Best model saved to {model_fp}")