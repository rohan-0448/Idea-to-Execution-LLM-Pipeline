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
import torch.distributions as dist # for distributional outputs
from sklearn.preprocessing import StandardScaler # for feature scaling

# init numin object

napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')
INPUT_SIZE = 47 # size of the input layer, here it is set equal to the number of features in the dataset
HIDDEN_SIZE = 100 # size of the hidden layer
OUTPUT_SIZE = 2  # Outputting mean and variance for a Gaussian distribution

# Define LSTM with Distributional Output (Gaussian)

class LSTMWithDistribution(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithDistribution, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)  # Output mean and variance

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # out shape: (batch_size, seq_len, hidden_size)
        out = self.linear(out[:, -1, :])  # Take the last time step's output
        # out shape: (batch_size, output_size) - mean and variance

        # Ensure variance is positive
        mean = out[:, 0]
        log_var = out[:, 1]  # Output log variance for numerical stability
        var = torch.exp(log_var)

        return mean, var

# Define Negative Log Likelihood Loss for Gaussian Distribution
def gaussian_nll_loss(mean, var, target):
    """
    Calculates the negative log-likelihood loss for a Gaussian distribution.

    Args:
        mean (torch.Tensor): Predicted mean of the Gaussian distribution.
        var (torch.Tensor): Predicted variance of the Gaussian distribution.
        target (torch.Tensor): True target values.

    Returns:
        torch.Tensor: The negative log-likelihood loss.
    """
    dist = torch.distributions.Normal(mean, torch.sqrt(var))
    log_prob = dist.log_prob(target)
    return -torch.mean(log_prob)


# Load the model (adjust path as needed)
lstm_dist = LSTMWithDistribution(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
lstm_dist.load_state_dict(torch.load('./saved_models/lstm_dist.pth'))  # Make sure the correct path is here
lstm_dist.eval()

# Load the scaler
scaler = StandardScaler()  # Instantiate scaler
#scaler = joblib.load('./saved_models/scaler.pkl') #Load the scaler



# code for inference and submission

WAIT_INTERVAL = 5 # time to wait before checking the round number again
NUM_ROUNDS = 25 # number of rounds to test

# create a temp directory as given in examples file

os.makedirs('API_submission_temp', exist_ok=True) # create a directory to store the predictions

# testing loop


def run_inference(test_data, curr_round, scaler):
    """Run inference using the trained LSTM to predict target_10 (distributionally)"""

    predictions = [] # init empty list to store predictions

    for ticker in test_data['id'].unique(): # iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy() # get data for the current ticker

        ticker_data.drop('id', axis=1, inplace=True) # drop the id column


        #Ensure correct columns are present
        #ticker_data = ticker_data[df.columns.tolist()] # Use the same columns as training data
        ticker_data = ticker_data.iloc[-1, :-2].values.tolist() # get the latest row from the test data

        #Scale the data
        ticker_data = np.array(ticker_data).reshape(1, -1)
        ticker_data = scaler.transform(ticker_data)


        with torch.no_grad():
            features = torch.tensor(ticker_data, dtype=torch.float32).unsqueeze(0) # convert the data to a tensor, add batch dimension
            mean, var = lstm_dist(features) # get the mean and variance from the model
            predicted_mean = mean.item() # extract the predicted mean
            predicted_variance = var.item()  # Extract predicted variance


            predictions.append({
                'id': ticker,
                'predicted_mean': predicted_mean,
                'predicted_variance': predicted_variance,
                'round_no': int(curr_round)
            }) # append prediction to the list
    
    return pd.DataFrame(predictions) # return the predictions as a dataframe


previous_round = None
rounds_completed = 0

while rounds_completed < NUM_ROUNDS: # run loop for all rounds
    try:
        curr_round = napi.get_current_round() # get current round

        if isinstance(curr_round, dict) and 'error' in curr_round: # check for error in getting current round
            print(f"Error getting round number: {curr_round['error']}")
            time.sleep(WAIT_INTERVAL)
            continue

        print(f"Current Round: {curr_round}")

        # check if round has changed
        if curr_round != previous_round: # check if the round has changed before submission

            # download round data
            print('Downloading round data...')  
            round_data = napi.get_data(data_type='round') # download round data
            
            if isinstance(round_data, dict) and 'error' in round_data: # check for error in downloading round data
                print(f"Failed to download round data: {round_data['error']}")
                time.sleep(WAIT_INTERVAL)
                continue
            
            # Convert round_data to DataFrame here to ensure it's a DataFrame
            round_data = pd.DataFrame(round_data)

            # Preprocess the data (scaling) - VERY IMPORTANT
            # First, ensure the scaler is fitted, even if loading it.  This prevents errors during transform.
            # Fit the scaler with all features from the first round or training data
            # scaler.fit(round_data.drop(['id', 'target_nom'], axis=1)) #Dropping target to prevent leakage in real application
            
            # process data and run inference
            print("Running inference...")
            output_df = run_inference(round_data, curr_round, scaler) # call run inference method on the test data           

            temp_csv_path = 'API_submission_temp/predictions.csv' 
            output_df.to_csv(temp_csv_path, index=False) # dump predictions to a csv

            # submit predictions
            print('Submitting predictions...')
            submission_response = napi.submit_predictions(temp_csv_path) # submit predictions using the API call

            if isinstance(submission_response, dict) and 'error' in submission_response: # check for errors in submitting response to the Numin platform
                print(f"Failed to submit predictions: {submission_response['error']}")
            else:
                print('Predictions submitted successfully...')
                rounds_completed += 1
                previous_round = curr_round

        else:
            print('Waiting for next round...') 

        time.sleep(WAIT_INTERVAL) # wait before checking if the round has changed
    
    except Exception as e: # check for any errors
        print(f"An error occurred: {str(e)}")
        time.sleep(WAIT_INTERVAL)

print('\nTesting complete!') # print message when testing is complete