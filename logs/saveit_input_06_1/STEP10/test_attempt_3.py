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
import random
import joblib
import io

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# init numin object

napi = NuminAPI(api_key= '946196ea-d7ad-6e6a-0854-0b45b15eaa4a')
INPUT_SIZE = 47 # size of the input layer, here it is set equal to the number of features in the dataset
HIDDEN_SIZE = 100 # size of the hidden layer
OUTPUT_SIZE = 5 # size of the output layer

# Define Expert Models (Example - Logistic Regression)
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Output is a single probability

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Define a simple feedforward network for experts
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output is a single suitability score

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return torch.sigmoid(out)  # Use sigmoid to get a probability-like score


# Define the MCTD component (Simplified for demonstration)
class MCTD:
    def __init__(self, num_trajectories=10, expert_threshold=0.5):
        self.num_trajectories = num_trajectories
        self.expert_threshold = expert_threshold

    def generate_trajectories(self, market_state, experts):
        # Generate random trajectories (replace with diffusion model later)
        trajectories = []
        for _ in range(self.num_trajectories):
            #Create random actions
            actions = [random.randint(0, 4) for _ in range(5)]
            trajectories.append(actions)  # Simplified: just a list of random actions
        return trajectories

    def evaluate_trajectories(self, trajectories, market_state, expert_opinions):
        # Evaluate trajectories based on expert opinions
        trajectory_scores = []
        for trajectory in trajectories:
            score = 0
            for expert_name, opinion in expert_opinions.items():
                if opinion > self.expert_threshold:  # Expert participates
                    # Simplified: Assume each action in the trajectory yields a small reward/penalty
                    score += sum(trajectory) * opinion  # Weighted sum of actions by expert opinion
            trajectory_scores.append(score)
        return trajectory_scores

    def select_best_trajectory(self, trajectories, trajectory_scores):
        # Select the best trajectory based on scores
        best_index = np.argmax(trajectory_scores)
        return trajectories[best_index]  # Return the actions of the best trajectory


# Define the Autonomy of Experts (AoE) - Example with 3 experts
class AutonomyOfExperts:
    def __init__(self, input_size, device):
        self.experts = {
            "trend_follower": FeedForwardNetwork(input_size, 32).to(device),
            "mean_reverter": FeedForwardNetwork(input_size, 32).to(device),
            "volatility_expert": FeedForwardNetwork(input_size, 32).to(device),
        }
        self.device = device
        # Load pre-trained expert models (replace with your actual paths)

        # Construct the absolute path to the saved_models directory
        saved_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['file_paths']['expert_models_dir']) # Corrected path

        for name, expert in self.experts.items():
            model_path = os.path.join(saved_models_dir, f"{name}.pth")
            try:
                expert.load_state_dict(torch.load(model_path, map_location=device))
                expert.eval()  # Set to evaluation mode
            except FileNotFoundError as e:
                print(f"Error loading model for {name}: {e}")
                raise  # Re-raise the exception to stop execution

    def evaluate_experts(self, market_state):
        """Evaluates all experts and returns their opinions."""
        expert_opinions = {}
        for name, expert in self.experts.items():
            with torch.no_grad():  # Disable gradients for evaluation
                market_state = market_state.to(self.device)  # Ensure data is on the correct device
                opinion = expert(market_state).item()  # Get the expert's opinion
                expert_opinions[name] = opinion
        return expert_opinions


# Define MLP class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # Input linear layer with dimensions input_size to hidden_size
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Linear layer with dimensions hidden_size to hidden_size
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.l3 = nn.Linear(hidden_size, output_size)  # Linear layer with dimensions hidden_size to output_size
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer

    def forward(self, X):
        out = self.l1(X)  # First linear layer
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the first linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l2(X)  # Apply second linear layer
        out = self.bn2(out)  # Batch normalization
        out = self.relu(out)  # Apply activation function to outputs of the second linear layer
        out = self.dropout(out)  # Apply dropout
        out = self.l3(X)  # Apply third linear layer
        return out

# Initialize components
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aoe = AutonomyOfExperts(INPUT_SIZE, DEVICE)  # Initialize experts
mctd = MCTD()  # Initialize MCTD


# load mlp.pth model from saved_models directory

# Initialize the model
mlp = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # Instantiate the MLP model
mlp.to(DEVICE)  # Ensure the model is on the right device (GPU or CPU)
mlp.load_state_dict(torch.load(config['file_paths']['output_directory']+'/mlp.pth', map_location=DEVICE))  # Load the model weights
mlp.eval()  # Set the model to eval mode

# Load the scaler
scaler = joblib.load(config['file_paths']['scaler_path'])

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
    """Run inference using the AoE + MCTD to predict target_10"""
    predictions = []  # Initialize empty list to store predictions

    for ticker in test_data['id'].unique():  # Iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy()  # Get data for the current ticker
        ticker_data.drop('id', axis=1, inplace=True)  # Drop the id column

        # Handle difference in input size
        if len(ticker_data.columns) < INPUT_SIZE:
            # ticker_data = ticker_data.iloc[:, :-2]
            padding_size = INPUT_SIZE - len(ticker_data.columns)
            padding = np.zeros((len(ticker_data), padding_size))
            ticker_data = pd.concat([ticker_data, pd.DataFrame(padding, index=ticker_data.index)], axis=1)  # pad with zeros
            # ticker_data = pd.concat([ticker_data, labels], axis=1)  # concatenate labels to the end
        elif len(ticker_data.columns) > INPUT_SIZE:
            ticker_data = ticker_data.iloc[:, :INPUT_SIZE]  # Truncate extra columns
            # ticker_data = ticker_data.iloc[:, :INPUT_SIZE] + ticker_data.iloc[:, -2:] #This line is wrong

        # Prepare data for inference
        features = ticker_data.iloc[-1].values  # Last timestep features
        
        # Ensure the features are reshaped and scaled only if they are not all NaNs
        if not np.isnan(features).all():
            features = scaler.transform(features.reshape(1, -1))  # Scale the features
        else:
            features = np.zeros((1, INPUT_SIZE))  # Use a zero vector if all NaNs

        features = torch.tensor(features, dtype=torch.float32).to(DEVICE)  # Convert to tensor and move to device

        # Get MLP prediction
        with torch.no_grad():
            mlp_output = mlp(features)
            predicted_class = torch.argmax(mlp_output).item()  # Get the predicted class

        # Expert Evaluation
        expert_opinions = aoe.evaluate_experts(features)

        # MCTD Trajectory Generation
        trajectories = mctd.generate_trajectories(features, aoe.experts)

        # MCTD Trajectory Evaluation
        trajectory_scores = mctd.evaluate_trajectories(trajectories, features, expert_opinions)

        # Select Best Trajectory
        best_trajectory = mctd.select_best_trajectory(trajectories, trajectory_scores)

        # Get Trading Action from Best Trajectory (assuming it's the first action)
        trading_action = best_trajectory[0] if best_trajectory else 0  # Default to 0 if no trajectory

        predictions.append({
            'id': ticker,
            'predictions': trading_action,  # Use the trading action as the prediction
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
            if isinstance(round_data, bytes):
                round_data = pd.read_csv(io.BytesIO(round_data))

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