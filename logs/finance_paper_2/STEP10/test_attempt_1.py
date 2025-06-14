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

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# init numin object

napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# Model parameters (adjust based on your actual model)
INPUT_SIZE = 47  # Adjust to the number of input features including embeddings
HIDDEN_SIZE = 128 # Example hidden size
OUTPUT_SIZE = 5   # 5 classes for the transformed target variable
EMBEDDING_DIM = 32 # Example embedding dimension for stock IDs
NUM_ASSETS = 100 #Example number of assets

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the EIIE network with LSTM IIEs and Attention Mechanism
class EIIE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, num_assets):
        super(EIIE, self).__init__()
        self.embedding = nn.Embedding(num_assets, embedding_dim) # Embedding layer for stock IDs
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # LSTM layer
        self.attention = nn.Linear(hidden_size, 1) # Attention mechanism
        self.linear = nn.Linear(hidden_size, output_size) # Output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, asset_ids):
        # x: (batch_size, seq_len, input_size) - numerical features
        # asset_ids: (batch_size) - LongTensor of asset IDs

        # Embedding layer for asset ids
        embedded = self.embedding(asset_ids)  # (batch_size, embedding_dim)
        
        # Expand the embedding to match the sequence length of numerical features
        embedded = embedded.unsqueeze(1).expand(-1, x.size(1), -1) # (batch_size, seq_len, embedding_dim)
       
        # Concatenate the embedded stock ID with the numerical features
        x = torch.cat((x, embedded), dim=2) # (batch_size, seq_len, input_size + embedding_dim)

        # LSTM layer
        lstm_out, _ = self.lstm(x) # (batch_size, seq_len, hidden_size)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(2), dim=1) # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1) # (batch_size, hidden_size)

        # Output layer
        output = self.linear(context_vector) # (batch_size, output_size)
        output = self.softmax(output) # (batch_size, output_size)

        return output


# Load the model
# eie = EIIE(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EMBEDDING_DIM, NUM_ASSETS).to(DEVICE)  # Instantiate the EIIE model
from training_code import MLP
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE) # Instantiate the MLP model

try:
    mlp.load_state_dict(torch.load(os.path.join(config['file_paths']['output_directory'], 'best_mlp.pth'), map_location=DEVICE))  # Load the model weights
except FileNotFoundError:
    print(f"Error: Model file not found at {os.path.join(config['file_paths']['output_directory'], 'best_mlp.pth')}")
    raise #Re-raise exception to stop execution
mlp.eval()  # Set the model to eval mode


# Function to preprocess input data
def preprocess_data(data):
    # Normalize numerical features and handle missing values
    numerical_cols = data.select_dtypes(include=np.number).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    for col in numerical_cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()  # Standardize
    return data

# Function to map stock IDs to embeddings
def map_stock_ids(stock_ids):
    #Create a mapping for the ids, this should be the same as the training
    unique_ids = pd.read_csv(config['file_paths']['stock_ids'])
    mapping = {stock_id: i for i, stock_id in enumerate(unique_ids['stock_ids'].unique())}
    return torch.tensor([mapping.get(stock_id, -1) for stock_id in stock_ids], dtype=torch.long).to(DEVICE) # Return tensor of mapped IDs


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
    """Run inference using the trained EIIE to predict target_10"""
    predictions = []  # Initialize empty list to store predictions

    for ticker in test_data['id'].unique():  # Iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy()  # Get data for the current ticker
        
        #Preprocess data
        ticker_data = preprocess_data(ticker_data)
        
        # Extract the stock ID and map it to an embedding
        asset_ids = map_stock_ids([ticker]) #Map the stock ids


        # Drop the 'id' column before feature processing
        ticker_data = ticker_data.drop('id', axis=1, errors='ignore')  # Drop the id column

        #Check for the target
        if 'target_10' in ticker_data.columns and 'target_0' in ticker_data.columns:
            labels = ticker_data[['target_10','target_0']]
            ticker_data = ticker_data.drop(['target_10','target_0'], axis=1, errors='ignore')

        # Pad or trim features to match input size
        num_features = len(ticker_data.columns)

        if num_features < INPUT_SIZE - EMBEDDING_DIM:
            padding_size = (INPUT_SIZE - EMBEDDING_DIM) - num_features
            ticker_data = pd.concat([ticker_data, pd.DataFrame(np.zeros((len(ticker_data), padding_size)))], axis=1)
        elif num_features > INPUT_SIZE - EMBEDDING_DIM:
            ticker_data = ticker_data.iloc[:, :(INPUT_SIZE - EMBEDDING_DIM)]

        # Prepare data for inference
        features = ticker_data.iloc[-1].values  # Last timestep features
        features = torch.tensor(features[np.newaxis, np.newaxis, :], dtype=torch.float32).to(DEVICE)  # Convert to tensor and move to DEVICE, add sequence length dimension


        with torch.no_grad():
            # output = eie(features, asset_ids)  # Get model predictions, pass the asset_ids
            output = mlp(features.squeeze()) # Get model predictions
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