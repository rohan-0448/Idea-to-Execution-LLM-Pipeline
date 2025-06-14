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

# Model parameters (adjust based on your training setup)
if 'model_params' not in config:
    print("Error: 'model_params' not found in config.yaml. Please add it.")
    exit()

NUM_ASSETS = config['model_params']['num_assets']  # Number of assets in the portfolio
INPUT_SIZE = config['model_params']['input_size']  # Size of the input feature vector
LSTM_HIDDEN_SIZE = config['model_params']['lstm_hidden_size']  # Hidden size for LSTM layers
CNN_KERNEL_SIZE = config['model_params']['cnn_kernel_size'] # Kernel size for CNN layers
PVM_SIZE = config['model_params']['pvm_size']

# Define the Actor-Critic model with EIIE and PVM
class ActorCriticEIIE(nn.Module):
    def __init__(self, num_assets, input_size, lstm_hidden_size, cnn_kernel_size, pvm_size):
        super(ActorCriticEIIE, self).__init__()
        self.num_assets = num_assets
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_kernel_size = cnn_kernel_size
        self.pvm_size = pvm_size

        # Ensemble of Identical Independent Evaluators (EIIE)
        self.iie_lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True) # LSTM for each asset
        self.iie_cnn = nn.Conv1d(lstm_hidden_size, 1, kernel_size=cnn_kernel_size) # CNN to extract spatial features

        # Actor Network
        self.actor_fc1 = nn.Linear(num_assets + pvm_size, 64)
        self.actor_fc2 = nn.Linear(64, num_assets)

        # Critic Network
        self.critic_fc1 = nn.Linear(num_assets + pvm_size, 64)
        self.critic_fc2 = nn.Linear(64, 1)  # Value function

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1) # Softmax to get portfolio weights
        self.tanh = nn.Tanh()

        # Portfolio Vector Memory (PVM)
        self.pvm = torch.zeros(pvm_size) # Initialize PVM

    def forward(self, x, prev_portfolio_vector):
        # x: (batch_size, num_assets, time_steps, input_size)

        # EIIE processing
        iie_outputs = []
        for i in range(self.num_assets):
            asset_data = x[:, i, :, :] # (batch_size, time_steps, input_size)
            lstm_out, _ = self.iie_lstm(asset_data) # (batch_size, time_steps, lstm_hidden_size)
            cnn_input = lstm_out.transpose(1, 2) # (batch_size, lstm_hidden_size, time_steps)
            cnn_out = self.iie_cnn(cnn_input).squeeze(1) # (batch_size, time_steps)
            iie_outputs.append(cnn_out[:, -1]) # Take the last time step output

        # Concatenate EIIE outputs
        eie_concat = torch.stack(iie_outputs, dim=1) # (batch_size, num_assets)

        # Actor Network
        actor_input = torch.cat([eie_concat, prev_portfolio_vector], dim=-1) # (batch_size, num_assets + pvm_size)
        actor_out = self.relu(self.actor_fc1(actor_input))
        portfolio_weights = self.softmax(self.actor_fc2(actor_out)) # (batch_size, num_assets)

        # Critic Network
        critic_input = torch.cat([eie_concat, prev_portfolio_vector], dim=-1) # (batch_size, num_assets + pvm_size)
        critic_out = self.relu(self.critic_fc1(critic_input))
        value = self.critic_fc2(critic_out) # (batch_size, 1)

        return portfolio_weights, value


# Load the trained Actor-Critic model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = ActorCriticEIIE(NUM_ASSETS, INPUT_SIZE, LSTM_HIDDEN_SIZE, CNN_KERNEL_SIZE, PVM_SIZE).to(DEVICE)

# Load the model weights
model.load_state_dict(torch.load(config['file_paths']['output_directory']+'/actor_critic_eie.pth'))  # Load the model weights
model.eval()  # Set the model to evaluation mode

# Initialize Portfolio Vector Memory (PVM) - assuming it's part of the model state
initial_portfolio_vector = torch.zeros(NUM_ASSETS).to(DEVICE)

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
    """Run inference using the trained Actor-Critic EIIE model to predict portfolio weights"""
    predictions = []  # Initialize empty list to store predictions

    # Initialize the portfolio vector for this round
    current_portfolio_vector = initial_portfolio_vector.clone().detach()

    for t in range(len(test_data)):
        round_data = test_data.iloc[[t]]
        ticker = round_data['id'].values[0]

        # Prepare data for the model
        ticker_data = test_data[test_data['id'] == ticker].copy()
        ticker_data.drop('id', axis=1, inplace=True)

        # Handle missing values (imputation)
        ticker_data = ticker_data.fillna(0)  # Replace NaN with 0, consider a more sophisticated imputation

        # Feature Scaling (Normalization) - Example
        # Replace with your actual normalization logic
        #ticker_data = (ticker_data - ticker_data.mean()) / ticker_data.std()

        # Reshape data for the model (batch_size, num_assets, time_steps, input_size)
        # Assuming you have historical data for each asset
        num_assets = len(test_data['id'].unique())
        time_steps = 1  # Example, adjust as needed
        input_size = len(ticker_data.columns)

        #Ensure that the input size matches the expected input size
        if input_size != INPUT_SIZE:
            print(f"Warning: Input size mismatch. Expected {INPUT_SIZE}, got {input_size}")
            continue
        
        #Prepare the price tensor
        price_tensor = ticker_data.values.reshape(1, num_assets, time_steps, input_size)
        price_tensor = torch.tensor(price_tensor, dtype=torch.float32).to(DEVICE)

        # Ensure the previous portfolio vector is on the correct device
        current_portfolio_vector = current_portfolio_vector.to(DEVICE)

        # Make prediction
        with torch.no_grad():
            portfolio_weights, _ = model(price_tensor, current_portfolio_vector.unsqueeze(0))
            portfolio_weights = portfolio_weights.squeeze(0)

        # Convert portfolio weights to numpy array and store the prediction
        predicted_weights = portfolio_weights.cpu().numpy()

        predictions.append({
            'id': ticker,
            'predictions': predicted_weights.tolist(),  # Store the portfolio weights
            'round_no': int(curr_round)
        })

        # Update the portfolio vector memory (PVM) with the current portfolio weights
        current_portfolio_vector = portfolio_weights.clone().detach()

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