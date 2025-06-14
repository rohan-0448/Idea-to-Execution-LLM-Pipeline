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
from sklearn.model_selection import train_test_split  # For creating validation set


# init numin object
napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

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

# Separate features and labels BEFORE train/val split
X = df.iloc[:, :-2].values.tolist() # separate features out from the labels
y = df.iloc[:, -1].values.tolist() # store labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # Use sklearn for splitting


# define constants

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # set device to mps if available, otherwise use cpu
INPUT_SIZE = len(X_train[0]) # get input size from the data - use training data!
print(INPUT_SIZE)  
#OUTPUT_SIZE = 5 # number of output classes # NO output size for regression
HIDDEN_SIZE = 100 # size of the hidden layer
NUM_QUANTILES = 5 # Define the number of quantiles.  Must be odd to include the median.
QUANTILES = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).float().to(DEVICE) # Example quantiles


# define Dataset class

class NuminDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float).to(DEVICE) # features used for training
        self.labels = torch.tensor(labels, dtype=torch.float).to(DEVICE) # labels used for training - FLOAT FOR REGRESSION

    def __len__(self):
        return len(self.labels) # return the number of samples in the dataset
    
    def __getitem__(self, self, idx):
        sample = self.data[idx] # get sample at index 'idx'
        label = self.labels[idx] # get label at index 'idx'
        return sample, label # return sample and label


# define Quantile Regression Neural Network

class QRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_quantiles):
        super(QRNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # input linear layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # hidden linear layer
        self.l3 = nn.Linear(hidden_size, num_quantiles) # output layer - one output per quantile
        self.relu = nn.ReLU() # activation function

    def forward(self, X):
        out = self.l1(X) # first linear layer
        out = self.relu(out) # apply activation function
        out = self.l2(out) # second linear layer
        out = self.relu(out) # apply activation function
        out = self.l3(out) # output layer
        return out

     
# instantiate qrnn

qrnn = QRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_QUANTILES).to(DEVICE)

# instantiate dataset and dataloader

train_dataset = NuminDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True) # SHUFFLE DATALOADER

val_dataset = NuminDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # No need to shuffle validation data

# Quantile Loss Function

def quantile_loss(preds, target, quantiles):
    """
    Computes the quantile loss.

    Args:
        preds (torch.Tensor): Predicted quantiles, shape (batch_size, num_quantiles)
        target (torch.Tensor): True values, shape (batch_size,)
        quantiles (torch.Tensor): Quantile values, shape (num_quantiles,)

    Returns:
        torch.Tensor: Quantile loss.
    """
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    assert preds.size(1) == quantiles.size(0)

    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.mean(torch.max((q - 1) * errors, q * errors)))  # changed from abs to max

    loss = torch.sum(torch.stack(losses))
    return loss

# optimizer

optimizer = torch.optim.Adam(qrnn.parameters(), lr=0.005) # REDUCED LEARNING RATE


# training loop

NUM_EPOCHS = 30 # number of training epochs
n_steps = len(train_dataloader) # number of steps in each epoch

qrnn.train() # Ensure the model is in training mode

for epoch in tqdm(range(NUM_EPOCHS)): # iterate through the dataset for NUM_EPOCHS
    for i, (features, labels) in enumerate(train_dataloader):
    
        inputs = features.to(DEVICE) # features are sent as inputs to the device
        labels = labels.to(DEVICE) # labels are sent as inputs to the device

        # forward pass
        outputs = qrnn(inputs) # get model predictions.  Shape: (batch_size, num_quantiles)
        # expand labels to match the shape of the outputs.  Shape: (batch_size, num_quantiles)
        labels = labels.unsqueeze(1).expand_as(outputs)

        loss = quantile_loss(outputs, labels[:,0], QUANTILES) # calculate quantile loss - pass the 0th quantile labels for now

        # backward pass
        optimizer.zero_grad() # zero out the gradients
        loss.backward() # backpropagate the loss 
        optimizer.step() # update the weights
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    # Validation Loop
    qrnn.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        for val_features, val_labels in val_dataloader:
            val_inputs = val_features.to(DEVICE)
            val_labels = val_labels.to(DEVICE)

            val_outputs = qrnn(val_inputs)
            val_labels = val_labels.unsqueeze(1).expand_as(val_outputs)
            val_loss += quantile_loss(val_outputs, val_labels[:, 0], QUANTILES).item()

    val_loss /= len(val_dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')
    qrnn.train() # Return the model to training mode

# save model to saved_models directory

model_fp = './saved_models/qrnn.pth' # path to save the model
torch.save(qrnn.state_dict(), model_fp) # save the model
print(f"Model saved to {model_fp}") # print message to confirm model has been saved