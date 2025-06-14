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
from sklearn.preprocessing import StandardScaler

# Read config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Numin API object
napi = NuminAPI(api_key='946196ea-d7ad-6e6a-0854-0b45b15eaa4a')

# Define constants
HIDDEN_SIZE = 128  # Size of the hidden layer
OUTPUT_SIZE = 5  # Number of output classes
NUM_LAYERS = 4  # Number of LSTM layers
SEQ_LENGTH = 10  # Sequence length used during training

# --- Data Preprocessing ---
def preprocess_data(df, seq_length=10, scaler=None, fitting=False):
    """
    Preprocesses the data for LSTM, including feature scaling, sequence creation, and target transformation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        seq_length (int): Length of the sequence for LSTM.
        scaler (StandardScaler): Fitted scaler to use for transforming the data
        fitting (bool): A flag to determine whether to fit the scaler or not

    Returns:
        tuple: (X, y, scaler) where X is a NumPy array of sequences and y is a NumPy array of labels.
    """

    X, y = [], []
    scaler_created = False
    for id in df['id'].unique():
        data = df[df['id'] == id].drop('id', axis=1).copy()  # Create a copy to avoid modifying the original DataFrame

        # Drop rows with missing values specific to the current 'id'
        data.dropna(inplace=True)

        if len(data) < seq_length:
             print(f"Skipping id {id} due to insufficient data after dropping NaNs.")
             continue  # Skip to the next 'id'

        # Feature scaling (fit scaler only on training data)
        numerical_features = data.columns[:-1]  # Assuming last column is the target
        if scaler is None:
            scaler = StandardScaler()
            if fitting:
                data[numerical_features] = scaler.fit_transform(data[numerical_features])
                scaler_created = True
            else:
                data[numerical_features] = scaler.transform(data[numerical_features])
        else:
             data[numerical_features] = scaler.transform(data[numerical_features])

        # Create sequences
        x_temp, y_temp = create_sequences(data, seq_length)
        X.extend(x_temp)
        y.extend(y_temp)

    return np.array(X), np.array(y), scaler

def create_sequences(data, seq_length):
    """Convert the data into sequences of length `seq_length`."""
    x_list, y_list = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length), :-2].values  # Features (sequence)
        y = data.iloc[i + seq_length, -1]  # Label (next value)
        y = int(2 * (y + 1))  # Convert labels from [-1, 1] to [0, 4]
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


# Define LSTM class (Example - Adapt to your actual HRL/Attention model)
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

# Mock Expert Strategy (Replace with actual MCTD/CodeMonkeys expert)
def mock_expert_strategy(data):
    # Replace this with actual expert logic
    # This example returns a random action
    return np.random.randint(0, OUTPUT_SIZE)

# Autonomy of Experts (AoE) Selection Mechanism (Adapt this part)
def select_experts(market_state, experts):
    """
    Selects the best experts based on market conditions and expert self-assessment.
    This is a placeholder.  Implement your AoE selection logic here.
    """
    # Placeholder:  Randomly select an expert
    num_experts = len(experts)
    selected_expert_index = np.random.randint(0, num_experts)
    return experts[selected_expert_index]



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

# Testing loop
def run_inference(test_data, curr_round, scaler, input_size):
    """Run inference using the trained LSTM and AoE to predict target_10"""
    predictions = []  # Initialize empty list to store predictions

    # Define experts (replace with actual MCTD/CodeMonkeys experts)
    experts = [mock_expert_strategy]  # Example: Using the mock expert

    for ticker in test_data['id'].unique():  # Iterate through all unique tickers
        ticker_data = test_data[test_data['id'] == ticker].copy()  # Get data for the current ticker
        ticker_data.drop('id', axis=1, inplace=True)  # Drop the id column

        # Prepare sequence for inference
        if len(ticker_data) >= SEQ_LENGTH:

            # Handle difference in input size
            #Remove the input size handling as it is being addressed in the model init now
            sequence = ticker_data.iloc[-SEQ_LENGTH:, :-2].values  # Last SEQ_LENGTH timesteps
            sequence = torch.tensor(sequence[np.newaxis, :], dtype=torch.float32).to(DEVICE)  # Convert to tensor and move to the selected device (GPU or CPU)

            # Market State (Adapt to your state representation)
            market_state = sequence  # Example: Use the sequence as the market state
            
            # AoE Expert Selection
            selected_expert = select_experts(market_state, experts)

            # Get prediction from selected expert
            predicted_class = selected_expert(ticker_data)

            predictions.append({
                'id': ticker,
                'predictions': predicted_class,
                'round_no': int(curr_round)
            })

    return pd.DataFrame(predictions)  # Return the predictions as a dataframe

# Load data
training_data_fp = config['file_paths']['training_data']  # Path to where the data is stored
df = pd.read_csv(training_data_fp)  # Read data into a pandas dataframe

# Preprocess the training data to retrieve the scaler object
X, y, scaler = preprocess_data(df, SEQ_LENGTH, fitting = True)

# Determine input size after preprocessing
INPUT_SIZE = X.shape[2]


# Initialize the model and set the device
lstm = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, layer_dim=NUM_LAYERS, output_dim=OUTPUT_SIZE)  # Instantiate LSTM model

# Set the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
lstm.to(DEVICE)  # Ensure the model is on the correct device (GPU or CPU)

# Load LSTM model from saved_models directory
lstm.load_state_dict(torch.load(config['file_paths']['output_directory']+'/lstm.pth'))  # Load the model weights
lstm.eval()  # Set the model to evaluation mode


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

            #call the preprocess data using the scaler we fit earlier
            X_test, y_test, _ = preprocess_data(round_data, SEQ_LENGTH, scaler = scaler, fitting = False) #No need to fit the scaler in the testing data
            
            # Process data and run inference
            print("Running inference...")
            round_data = round_data[round_data['id'].isin(df['id'].unique())] #filter to avoid future size mismatch error
            output_df = run_inference(round_data, curr_round, scaler, INPUT_SIZE)  # Call run_inference method on the test data

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