# Import libraries
from numin import NuminAPI  # Library to download data and make submissions to the Numin platform
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computation library
from tqdm import tqdm  # Progress bar library
import torch  # Deep learning library
import torch.nn as nn  # Neural network library
import os  # To access files and directories
import time  # To measure time
from torch.utils.data import Dataset, DataLoader  # To create custom datasets and dataloaders
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Scalers
from sklearn.impute import SimpleImputer  # Imputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read config file
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError as e:
    logging.error(f"Error: Config file not found: {e}")
    exit()
except yaml.YAMLError as e:
    logging.error(f"Error parsing config file: {e}")
    exit()
except Exception as e:
    logging.error(f"Error loading config: {e}")
    exit()

# Initialize Numin API object
try:
    api_key = config.get('api_key')
    if api_key is None:
        raise ValueError("API key not found in config.yaml")
    napi = NuminAPI(api_key=api_key)
except Exception as e:
    logging.error(f"Error initializing Numin API: {e}")
    exit()

# Define constants
INPUT_SIZE = 47  # Number of features per timestep
HIDDEN_SIZE = 128  # Size of the hidden layer
OUTPUT_SIZE = 5  # Number of output classes
NUM_LAYERS = 4  # Number of LSTM layers
SEQ_LENGTH = 10  # Sequence length used during training

# Define LSTM class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model and set the device
lstm = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layer_dim=NUM_LAYERS, output_dim=OUTPUT_SIZE)  # Instantiate LSTM model

# Set the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
lstm.to(DEVICE)  # Ensure the model is on the correct device (GPU or CPU)

# Load LSTM model from saved_models directory
try:
    model_path = config['file_paths']['output_directory']+'/best_lstm.pth'
    lstm.load_state_dict(torch.load(model_path))  # Load the model weights
    lstm.eval()  # Set the model to evaluation mode
    logging.info("Successfully loaded the trained model from %s.", model_path)

except FileNotFoundError as e:
    logging.error(f"Error: Model file not found: {e}")
    exit()  # Exit if model loading fails
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    exit()

# Load imputation model and scaler
try:
    imputer_path = config['file_paths']['output_directory']+'/imputer.pth'
    scaler_path = config['file_paths']['output_directory']+'/scaler.pth'

    imputer = torch.load(imputer_path)
    scaler = torch.load(scaler_path)
    logging.info("Successfully loaded Imputer and Scaler")

except FileNotFoundError as e:
    logging.error(f"Error: Imputer/Scaler file not found: {e}")
    exit()  # Exit if imputer loading fails
except Exception as e:
    logging.error(f"Error loading the Imputer/Scaler: {e}")
    exit()

# Code for inference and submission
WAIT_INTERVAL = 5  # Time to wait before checking the round number again
NUM_ROUNDS = 25  # Number of rounds to test

# Create a directory to store the predictions
os.makedirs(os.path.dirname(config['file_paths']['temp_numin_results_path']), exist_ok=True)

# Create or overwrite the predictions file
predictions_file = os.path.join(config['file_paths']['temp_numin_results_path'], 'predictions.csv')
if os.path.exists(predictions_file):
    os.remove(predictions_file)  # Remove existing file if it exists
with open(predictions_file, 'w') as f:
    f.write('')  # Create empty file

def preprocess_testing_data(testing_data, imputer, scaler):
    """Preprocesses testing data using the fitted imputation model and scaler."""
    df = testing_data.copy()
    #df = df.drop('id', axis=1) #Moved the drop statement into the run_inference function

    # Impute missing values
    df = imputer.transform(df)

    # Scale numerical features
    df = scaler.transform(df)

    return df

# Testing loop
def run_inference(test_data, curr_round, imputer, scaler):
    """Run inference using the trained LSTM to predict target_10"""
    predictions = []  # Initialize empty list to store predictions

    for ticker in test_data['id'].unique():  # Iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy()  # Get data for the current ticker
        
        # Preprocess the data
        try:
            ticker_data_processed = preprocess_testing_data(ticker_data.drop(['id', 'score_10', 'score_1'], axis=1), imputer, scaler)
        except Exception as e:
             logging.error(f"Preprocessing error for ticker {ticker}: {e}")
             continue


        # Prepare sequence for inference
        if ticker_data_processed.shape[0] >= SEQ_LENGTH:
            sequence = ticker_data_processed[-SEQ_LENGTH:]
            sequence = torch.tensor(sequence[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            logging.debug(f"Sequence shape: {sequence.shape}, dtype: {sequence.dtype}")

            try:
                with torch.no_grad():
                    output = lstm(sequence)  # Get model predictions
                    predicted_class = int(torch.argmax(output))  # Get the predicted class

                    predictions.append({
                        'id': ticker,
                        'predictions': predicted_class,
                        'round_no': int(curr_round)
                    })
            except Exception as e:
                logging.error(f"Inference error for ticker {ticker}: {e}")
                continue

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
            output_df = run_inference(round_data, curr_round, imputer, scaler)  # Call run_inference method on the test data

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