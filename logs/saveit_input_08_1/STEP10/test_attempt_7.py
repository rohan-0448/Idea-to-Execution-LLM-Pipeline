import os
import yaml
import joblib
import torch
import torch.nn as nn
from numin import NuminAPI
import pandas as pd
import numpy as np
import random
import time

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# init numin object

napi = NuminAPI(api_key= config['api_key']) # Replace with your API Key
INPUT_SIZE = 47 # size of the input layer, here it is set equal to the number of features in the dataset
HIDDEN_SIZE = 100 # size of the hidden layer
OUTPUT_SIZE = 5 # size of the output layer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# define MLP

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # input linear layer with dimensions input_size to hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size) # linear with dimensions hidden_size to hidden_size 
        self.l3 = nn.Linear(hidden_size, output_size) # linear layer with dimension hidden_size to output_size 
        self.relu = nn.ReLU() # activation function

    def forward(self, X):
        out = self.l1(X) # first linear layer
        out = self.relu(out) # apply activation function to outputs of the first linear layer
        out = self.l2(X) # apply second linear layer
        out = self.relu(out) # apply activation function to outputs of the second linear layer
        out = self.l3(out) # apply third linear layer

        return out

# load mlp.pth model from saved_models directory

# Construct the model path correctly
output_dir = config['file_paths']['output_directory']
model_filename = 'best_mlp.pth'
model_path = os.path.join(output_dir, model_filename)

# Ensure the path is absolute AND normalized
if not os.path.isabs(model_path):
    model_path = os.path.join(os.getcwd(), model_path)

model_path = os.path.normpath(model_path) # Normalize the path (convert slashes to OS-specific ones, remove redundant separators)

print(f"Attempting to load model from: {model_path}") # Print the path for debugging

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE) # instantiate mlp model with input size, hidden size and output size
mlp.load_state_dict(torch.load(model_path, map_location=DEVICE)) # load the model weights
mlp.eval() # set the model to eval mode

# Load the scaler
scaler_path = os.path.join(config['file_paths']['output_directory'], 'scaler.pkl')

# Ensure the path is absolute
if not os.path.isabs(scaler_path):
    scaler_path = os.path.join(os.getcwd(), scaler_path)

scaler_path = os.path.normpath(scaler_path)

print(f"Attempting to load scaler from: {scaler_path}")  # Print the path for debugging
scaler = joblib.load(scaler_path)
     
# code for inference and submission

WAIT_INTERVAL = 5 # time to wait before checking the round number again
NUM_ROUNDS = 25 # number of rounds to test

# Create a directory to store the predictions
os.makedirs(os.path.dirname(config['file_paths']['temp_numin_results_path']), exist_ok=True)

# Create or overwrite the predictions file
predictions_file = os.path.join(config['file_paths']['temp_numin_results_path'], 'predictions.csv')
if os.path.exists(predictions_file):
    os.remove(predictions_file)  # Remove existing file if it exists
with open(predictions_file, 'w') as f:
    f.write('')  # Create empty file

# Testing loop
def run_inference(test_data, curr_round):
    """Run inference using the trained MLP to predict target_10"""
    predictions = []  # Initialize empty list to store predictions

    for ticker in test_data['id'].unique():  # Iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy()  # Get data for the current ticker
        ticker_data.drop('id', axis=1, inplace=True)  # Drop the id column

        # Handle difference in input size
        if len(ticker_data.columns) < INPUT_SIZE:
            labels = ticker_data.iloc[:, -2:]  # separate out labels
            ticker_data = ticker_data.iloc[:, :-2]
            padding_size = INPUT_SIZE - len(ticker_data.columns)
            print(f"Warning: Ticker {ticker} has {len(ticker_data.columns)} features, padding with {padding_size} zeros.")
            ticker_data = pd.concat([ticker_data, pd.DataFrame(np.zeros((len(ticker_data), padding_size)))], axis=1)  # pad with zeros
            ticker_data = pd.concat([ticker_data, labels], axis=1)  # concatenate labels to the end
        elif len(ticker_data.columns) > INPUT_SIZE:
            ticker_data = ticker_data.iloc[:, :INPUT_SIZE] # Truncate extra columns
            # Potentially add or average with labels, depending on the columns' meanings
            # ticker_data = ticker_data.iloc[:, :INPUT_SIZE] + ticker_data.iloc[:, -2:] # This might give incorrect results! Be sure!


        # Prepare data for inference
        features = ticker_data.iloc[-1, :-2].values  # Last timestep features

        # Scale the features using the loaded scaler
        features = scaler.transform(features.reshape(1, -1))

        features = torch.tensor(features, dtype=torch.float32).to(DEVICE)  # Convert to tensor

        with torch.no_grad():
            output = mlp(features)  # Get model predictions
            predicted_class = int(torch.argmax(output))  # Get the predicted class

            predictions.append({
                'id': ticker,
                'predictions': predicted_class,
                'round_no': int(curr_round)
            })

    return pd.DataFrame(predictions)  # Return the predictions as a dataframe

previous_round = None
rounds_completed = 0

while rounds_completed < NUM_ROUNDS:  # Run loop for all rounds
    try:
        curr_round = napi.get_current_round()  # Get current round

        if isinstance(curr_round, dict) and 'error' in curr_round:  # Check for error in getting current round
            print(f"Error getting round number: {curr_round['error']}")
            time.sleep(WAIT_INTERVAL)
            continue

        print(f"Current Round: {curr_round}")

        # Check if round has changed
        if curr_round != previous_round:  # Check if the round has changed before submission
            # Download round data
            print('Downloading round data...')
            round_data = napi.get_data(data_type='round')  # Download round data

            if isinstance(round_data, dict) and 'error' in round_data:  # Check for error in downloading round data
                print(f"Failed to download round data: {round_data['error']}")
                time.sleep(WAIT_INTERVAL)
                continue

            # Process data and run inference
            print("Running inference...")
            output_df = run_inference(round_data, curr_round)  # Call run_inference method on the test data

            # Save predictions to a temporary CSV file
            output_df.to_csv(predictions_file, index=False)  # Dump predictions to a CSV

            # Submit predictions
            print('Submitting predictions...')
            submission_response = napi.submit_predictions(predictions_file)  # Submit predictions using the API call

            if isinstance(submission_response, dict) and 'error' in submission_response:  # Check for errors in submission
                print(f"Failed to submit predictions: {submission_response['error']}")
            else:
                print('Predictions submitted successfully...')
                rounds_completed += 1
                previous_round = curr_round

        else:
            print('Waiting for next round...')

        time.sleep(WAIT_INTERVAL)  # Wait before checking if the round has changed

    except Exception as e:  # Check for any errors
        print(f"An error occurred: {str(e)}")
        time.sleep(WAIT_INTERVAL)

print('\nTesting complete!')  # Print message when testing is complete