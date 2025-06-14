# import libraries

from numin import NuminAPI # library to download data and make submissions to the Numin platform
import pandas as pd # data manipulation library
import numpy as np # numerical computation library
from tqdm import tqdm # progress bar library
import torch # deep learning library
import torch.nn as nn # neural network library
import os # to acess files and directories
import time # to measure time
from torch.utils.data import Dataset, DataLoader # to create custom datsets and dataloaders
from torch.distributions import Normal # Import the Normal distribution
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# init numin object
# napi = NuminAPI(api_key= 'YOUR_API_KEY') # Replace with your actual API key

# download data, commented out if data already exists

# data = napi.get_data(data_type="training")  # BytesIO

# file_path = "training_data.zip"  # Change the file name as needed

# with open(file_path, 'wb') as f:
#     f.write(data.getbuffer())

# print(f"Data downloaded and saved to {file_path}")

# import data

training_data_fp = './training_data/df_val_31-May-2024.csv' # path to where the data is stored
df = pd.read_csv(training_data_fp) # read data into a pandas datframe
df = df.drop('id', axis=1) # drop the id column
df.dropna(inplace=True) # drop rows with missing values in the training data

X = df.iloc[:, :-2].values.tolist() # separate features out from the labels
y = df.iloc[:, -1].values.tolist() # store labels
# No conversion to classification. Keep as regression target.

# Data scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# define constants

DEVICE = torch.device('cpu') # set device to cpu
INPUT_SIZE = len(X[0]) # get input size from the data
print(INPUT_SIZE)
HIDDEN_SIZE = 100 # size of the hidden layer
OUTPUT_SIZE = 2  # Mean and standard deviation for Normal distribution



# define Dataset class

class NuminDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float) # features used for training
        self.labels = torch.tensor(labels, dtype=torch.float) # labels used for training, changed to float for regression

    def __len__(self):
        return len(self.labels) # return the number of samples in the dataset

    def __getitem__(self, idx):
        sample = self.data[idx] # get sample at index 'idx'
        label = self.labels[idx] # get label at index 'idx'
        return sample, label # return sample and label


# define LSTM with distributional output

class LSTMDistributionOutput(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMDistributionOutput, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)  # Output mean and stddev

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])  # shape: (batch_size, output_size)
        mean = out[:, 0] # Extract predicted mean
        std = torch.exp(out[:, 1]) # Extract predicted standard deviation, use exp to ensure positivity

        return mean, std

# Instantiate the model
lstm_dist = LSTMDistributionOutput(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

# instantiate dataset and dataloader

train_dataset = NuminDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Added shuffle

val_dataset = NuminDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False) # No shuffle for validation



# Loss function
def negative_log_likelihood(mean, std, target):
    """
    Calculates the negative log-likelihood loss for a normal distribution.

    Args:
        mean (torch.Tensor): Predicted mean of the normal distribution.
        std (torch.Tensor): Predicted standard deviation of the normal distribution.
        target (torch.Tensor): Actual target values.

    Returns:
        torch.Tensor: The negative log-likelihood loss.
    """
    dist = Normal(mean, std)
    log_prob = dist.log_prob(target)
    return -log_prob.mean()

# Optimizer
optimizer = torch.optim.Adam(lstm_dist.parameters(), lr=0.001)  # Reduced learning rate


# training loop

NUM_EPOCHS = 30 # number of training epochs
n_steps = len(train_dataloader) # number of steps in each epoch

lstm_dist.to(DEVICE) # Move model to device before training

for epoch in tqdm(range(NUM_EPOCHS)): # iterate through the dataset for NUM_EPOCHS

    lstm_dist.train() # Set model to training mode
    train_loss = 0.0
    for i, (features, labels) in enumerate(train_dataloader):

        inputs = features.unsqueeze(1).to(DEVICE) # features are sent as inputs, added unsqueeze for sequence dimension
        labels = labels.to(DEVICE)

        # forward pass
        mean, std = lstm_dist(inputs) # get model predictions
        loss = negative_log_likelihood(mean, std, labels) # calculate loss
        train_loss += loss.item()

        # backward pass
        optimizer.zero_grad() # zero out the gradients
        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights

    # Validation loop
    lstm_dist.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for features, labels in val_dataloader:
            inputs = features.unsqueeze(1).to(DEVICE)
            labels = labels.to(DEVICE)

            mean, std = lstm_dist(inputs)
            loss = negative_log_likelihood(mean, std, labels)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


# save model to saved_models directory

model_fp = './saved_models/lstm_dist.pth' # path to save the model
torch.save(lstm_dist.state_dict(), model_fp) # save the model
print(f"Model saved to {model_fp}") # print message to confirm model has been saved