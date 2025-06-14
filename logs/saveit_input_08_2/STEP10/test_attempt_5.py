# import libraries

from numin import NuminAPI # library to download data and make submissions to the numin platform
import pandas as pd # data manipulation library
import numpy as np # numerical computation library
from tqdm import tqdm # progress bar library
import torch # deep learning library
import torch.nn as nn # neural network library
import os # to access files and directories
import time # to measure time
from torch.utils.data import Dataset, DataLoader # to create custom datasets and dataloaders
import yaml
from sklearn.preprocessing import StandardScaler
#from sklearn.impute import SimpleImputer #Not needed
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# init numin object

napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a') #Replace with your API key
INPUT_SIZE = 47 # size of the input layer, here it is set equal to the number of features in the dataset
HIDDEN_SIZE = 100 # size of the hidden layer
OUTPUT_SIZE = 5 # size of the output layer

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

# Initialize the model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # Instantiate the MLP model

# Set the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
mlp.to(DEVICE)  # Ensure the model is on the right device (GPU or CPU)

# Load the best model (or final model, depending on your choice)
model_file_path = os.path.join(config['file_paths']['output_directory'], 'best_mlp.pth') #Or 'final_mlp.pth'

# Load scaler
scaler_fp = os.path.join(config['file_paths']['output_directory'], 'scaler.pkl')

# Check if the file exists
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file not found at: {model_file_path}.  Make sure training completed successfully and the output directory in config.yaml is correct.")

if not os.path.exists(scaler_fp):
    raise FileNotFoundError(f"Scaler file not found at: {scaler_fp}.  Make sure training completed successfully and the output directory in config.yaml is correct.")

# Load the scaler
try:
    with open(scaler_fp, 'rb') as f: # Use a context manager for file handling
        scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Scaler file not found at {scaler_fp}.  Ensure training was completed and the path is correct.")
    raise #Re-raise the exception to stop the script
except Exception as e: #Catch any other exception during loading
    print(f"Error loading scaler: {e}")
    raise

mlp.load_state_dict(torch.load(model_file_path, map_location=DEVICE))  # Load the model weights and map to the correct device
mlp.eval()  # Set the model to eval mode




def preprocess_data(df):
    """
    Preprocesses the input DataFrame: handles missing values and scales features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Impute missing values.  Fit during training and use the fitted imputer during testing

    # Scale numerical features.  Fit during training, and transform during testing.
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


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

        # Preprocess the ticker data (consistent with training)
        ticker_data = preprocess_data(ticker_data)


        # Handle difference in input size
        if len(ticker_data.columns) < INPUT_SIZE:
            labels = ticker_data.iloc[:, -2:]  # separate out labels
            ticker_data = ticker_data.iloc[:, :-2]
            ticker_data = pd.concat([ticker_data, pd.DataFrame(np.zeros((len(ticker_data), INPUT_SIZE - len(ticker_data.columns))))], axis=1)  # pad with zeros
            ticker_data = pd.concat([ticker_data, labels], axis=1)  # concatenate labels to the end
        elif len(ticker_data.columns) > INPUT_SIZE:
            ticker_data = ticker_data.iloc[:, :INPUT_SIZE] + ticker_data.iloc[:, -2:]

        # Prepare data for inference
        features = ticker_data.iloc[-1, :-2].values  # Last timestep features
        features = torch.tensor(features[np.newaxis, :], dtype=torch.float32).to(DEVICE)  # Convert to tensor and move to DEVICE

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