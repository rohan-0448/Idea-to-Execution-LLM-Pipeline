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
from sklearn.model_selection import train_test_split # to split training data into training and validation sets
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random


# init numin object
# napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a') #Commented out for now as it is not being used, replace with API key


# download data, commented out if data already exists

# data = napi.get_data(data_type="training")  # BytesIO

# file_path = "training_data.zip"  # Change the file name as needed

# with open(file_path, 'wb') as f:
#     f.write(data.getbuffer())

# print(f"Data downloaded and saved to {file_path}")

# import data
# Set random seeds for reproducibility
seed = 42  # Or any other integer
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


training_data_dir = './training_data'
training_data_fp = os.path.join(training_data_dir, 'df_val_31-May-2024.csv') # use os.path.join for robustness


# Check if the directory exists; if not, create it (optional but recommended)
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)
    print(f"Created directory: {training_data_dir}")

# Check if the file exists before attempting to read it
if os.path.exists(training_data_fp):
    try:
        df = pd.read_csv(training_data_fp)  # read data into a pandas DataFrame
        print(f"Data loaded successfully from: {training_data_fp}")

        #Data processing can now continue

    except FileNotFoundError:
        print(f"Error: The file {training_data_fp} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
else:
    print(f"Error: The file {training_data_fp} does not exist.  Make sure the data is downloaded or the path is correct.")
    exit() # stop execution if file not found


# Preprocessing
# Assuming df['id'] contains your stock IDs as strings
label_encoder = LabelEncoder()
df['id_encoded'] = label_encoder.fit_transform(df['id'])
df = df.drop('id', axis=1) # now drop the original id column

df.dropna(inplace=True) # drop rows with missing values in the training data


# Scale numerical features
numerical_features = df.columns[:-2]  # Assuming the last two columns are the target
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


X = df.iloc[:, :-2].values.tolist() # separate features out from the labels
y = df.iloc[:, -1].values.tolist() # store labels 
y = [int(2 * (score_10 + 1)) for score_10 in y] # convert labels from [-1,1] to [0,4] to convert into a classification problem

#Split the data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #Standard split with an 80/20 ratio

# define constants

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # set device to mps if available, else use cpu
INPUT_SIZE = len(X[0]) # get input size from the data
print(INPUT_SIZE)  
OUTPUT_SIZE = 5 # number of output classes
HIDDEN_SIZE = 100 # size of the hidden layer


# define Dataset class

class NuminDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float).to(DEVICE) # features used for training, moved to device
        self.labels = torch.tensor(labels, dtype=torch.long).to(DEVICE) # labels used for training, moved to device

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

    def forward(self, X):
        out = self.l1(X) # first linear layer
        out = self.relu(out) # apply activation function to outputs of the first linear layer
        out = self.l2(out) # apply second linear layer
        out = self.relu(out) # apply activation function to outputs of the second linear layer
        out = self.l3(out) # apply third linear layer

        return out

     
# instantiate mlp
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)  # Instantiate model and move to device

# instantiate dataset and dataloader

train_dataset = NuminDataset(X_train, y_train)
val_dataset = NuminDataset(X_val, y_val) #Create a validation dataset
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True) #Shuffle training data
val_dataloader = DataLoader(val_dataset, batch_size=4)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005) #Lowered learning rate to 0.005, was 0.05

# training loop

NUM_EPOCHS = 10 # number of training epochs, reduced from 30
n_steps = len(train_dataloader) # number of steps in each epoch

for epoch in tqdm(range(NUM_EPOCHS)): # iterate through the dataset for NUM_EPOCHS
    for i, (features, labels) in enumerate(train_dataloader):
    
        inputs = features # features are sent as inputs
        labels = labels

        # forward pass
        outputs = mlp(inputs) # get model predictions
        loss = criterion(outputs, labels) # calculate loss

        # backward pass
        optimizer.zero_grad() # zero out the gradients
        loss.backward() # backpropagate the loss 
        optimizer.step() # update the weights
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    #Validation Loop
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for features, labels in val_dataloader:
            outputs = mlp(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Accuracy: {(100 * correct / total):.2f}%')

# save model to saved_models directory

model_fp = './saved_models/mlp.pth' # path to save the model
torch.save(mlp.state_dict(), model_fp) # save the model
print(f"Model saved to {model_fp}") # print message to confirm model has been saved