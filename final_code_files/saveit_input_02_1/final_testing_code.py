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


# init numin object

napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')
INPUT_SIZE = 47 # size of the input layer, here it is set equal to the number of features in the dataset
HIDDEN_SIZE = 100 # size of the hidden layer
NUM_QUANTILES = 5  # Number of quantiles to predict

# define Quantile Regression Neural Network (QRNN)
class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_quantiles):
        super(QRNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_quantiles)  # Output layer predicts quantiles
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Quantile Loss Function
def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.mean(torch.max((q - 1) * errors, q * errors)))
    return torch.sum(torch.stack(losses))

# load qrnn.pth model from saved_models directory
qrnn = QRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_QUANTILES)
qrnn.load_state_dict(torch.load('./saved_models/qrnn.pth', map_location=torch.device('cpu')))  # Load model, map to CPU if no GPU
qrnn.eval()
     
# code for inference and submission

WAIT_INTERVAL = 5 # time to wait before checking the round number again
NUM_ROUNDS = 25 # number of rounds to test

# create a temp directory as given in examples file

os.makedirs('API_submission_temp', exist_ok=True) # create a directory to store the predictions

# testing loop


def run_inference(test_data, curr_round):
    """Run inference using the trained QRNN to predict quantiles for target_10"""

    predictions = [] # init empty list to store predictions

    for ticker in test_data['id'].unique(): # iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy() # get data for the current ticker

        ticker_data.drop('id', axis=1, inplace=True) # drop the id column

        # for column in ticker_data.columns:
        #     if column not in df.columns:
        #         ticker_data.drop(column, axis=1, inplace=True) # drop columns not present in the training data

        
        ticker_data = ticker_data.iloc[-1, :-2].values.tolist() # get the latest row from the test data

        with torch.no_grad():
            features = torch.tensor(ticker_data, dtype=torch.float32) # convert the data to a tensor, ensure correct dtype
            quantile_preds = qrnn(features) # get the quantile predictions

            # For this example, let's take the median (0.5 quantile) as the prediction
            # You can modify this to use other quantiles or a combination
            median_index = NUM_QUANTILES // 2
            output = quantile_preds[:, median_index].item()

            predictions.append({
                'id': ticker,
                'predictions': output,
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

            # process data and run inference
            print("Running inference...")
            output_df = run_inference(round_data, curr_round) # call run inference method on the test data           

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
```"