# import libraries

from numin import NuminAPI # library to download data and make submissions to the Numin platform
import pandas as pd # data manipulation library
import numpy as np # numerical computation library
from tqdm import tqdm # progress bar library
import torch # deep learning library
import torch.nn as nn # neural network library
import os # to acess files and directories
import time # to measure time
from torch.utils.data import Dataset, DataLoader, random_split # to create custom datsets and dataloaders
import yaml
from sklearn.preprocessing import StandardScaler # Feature scaling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# init numin object
napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# download data, commented out if data already exists

# data = napi.get_data(data_type="training")  # BytesIO

# file_path = "training_data.zip"  # Change the file name as needed

# with open(file_path, 'wb') as f:
#     f.write(data.getbuffer())

# print(f"Data downloaded and saved to {file_path}")

# import data

training_data_fp = config['file_paths']['training_data'] # path to where the data is stored
df = pd.read_csv(training_data_fp) # read data into a pandas datframe
df = df.drop('id', axis=1) # drop the id column
df.dropna(inplace=True) # drop rows with missing values in the training data

X = df.iloc[:, :-2].values.tolist() # separate features out from the labels
y = df.iloc[:, -1].values.tolist() # store labels 
y = [int(2 * (score_10 + 1)) for score_10 in y] # convert labels from [-1,1] to [0,4] to convert into a classification problem

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)


# define constants

DEVICE = torch.device('mps') # set device to cuda if available
INPUT_SIZE = len(X[0]) # get input size from the data
print(INPUT_SIZE)  
OUTPUT_SIZE = 5 # number of output classes
HIDDEN_SIZE = 100 # size of the hidden layer


# define Dataset class

class NuminDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float) # features used for training
        self.labels = torch.tensor(labels, dtype=torch.long) # labels used for training

    def __len__(self):
        return len(self.labels) # return the number of samples in the dataset
    
    def __getitem__(self, idx):
        sample = self.data[idx] # get sample at index 'idx'
        label = self.labels[idx] # get label at index 'idx'
        return sample, label # return sample and label


# define MLP

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # input linear layer with dimensions input_size to hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size) # linear layer with dimensions hidden_size to hidden_size
        self.l3 = nn.Linear(hidden_size, output_size) # linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU() # activation function
        self.dropout = nn.Dropout(p=0.2)  # Add dropout for regularization

    def forward(self, X):
        out = self.l1(X) # first linear layer
        out = self.relu(out) # apply activation function to outputs of the first linear layer
        out = self.dropout(out) # Apply dropout
        out = self.l2(out) # apply second linear layer
        out = self.relu(out) # apply activation function to outputs of the second linear layer
        out = self.dropout(out) # Apply dropout
        out = self.l3(out) # apply third linear layer
        return out

     
# instantiate mlp

mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

# instantiate dataset and dataloader

dataset = NuminDataset(X, y)

# Split data into training and validation sets (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-5) # Added weight decay for regularization

# training loop

NUM_EPOCHS = 30 # number of training epochs
n_steps = len(dataloader) # number of steps in each epoch

best_val_loss = float('inf') # Initialize best validation loss for model checkpointing

for epoch in tqdm(range(NUM_EPOCHS)): # iterate through the dataset for NUM_EPOCHS

    # Training Loop
    mlp.train()  # Set the model to training mode
    total_loss = 0
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
    
        # forward pass
        outputs = mlp(features) # get model predictions
        loss = criterion(outputs, labels) # calculate loss
        total_loss += loss.item()

        # backward pass
        optimizer.zero_grad() # zero out the gradients
        loss.backward() # backpropagate the loss 
        optimizer.step() # update the weights
    
    avg_train_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}')

    # Validation Loop
    mlp.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation during validation
        for features, labels in val_dataloader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = mlp(features)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    # Model Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_fp = config['file_paths']['output_directory'] + '/best_mlp.pth' # path to save the model
        torch.save(mlp.state_dict(), model_fp) # save the model
        print(f"Best model saved to {model_fp}") # print message to confirm model has been saved


# save model to saved_models directory

model_fp = config['file_paths']['output_directory'] + '/last_mlp.pth' # path to save the model
torch.save(mlp.state_dict(), model_fp) # save the model
print(f"Last model saved to {model_fp}") # print message to confirm model has been saved