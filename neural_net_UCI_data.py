from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset into a DataFrame to easily manage categorical transformations
data = pd.read_csv("adult.data", header=None)
columns = {1, 3, 5, 6, 7, 8, 9, 13}  # Set of indices with categorical data

# Initialize LabelEncoder dictionary
def initialize_encoders(data, columns):
    return {col: LabelEncoder().fit(data[col].astype(str)) for col in columns}

encoders = initialize_encoders(data, columns)

def parse_line(row, encoders):
    inputs = []
    # Use iloc to handle indexing properly with Series objects
    for i in range(len(row) - 1):  # Excluding the last column (output)
        token = row.iloc[i]
        if i in encoders:
            # Ensure the token is a string, as encoders expect that format
            inputs.append(float(encoders[i].transform([str(token)])[0]))
        else:
            inputs.append(float(token))
    # Directly access the last column for the output
    output = [1.0 if row.iloc[-1].strip() == '>50K' else 0.0]
    return (inputs, output)



def normalize(data):
    max_values = [max(col) for col in zip(*[d[0] for d in data])]  # Ensure normalization considers only input part
    min_values = [min(col) for col in zip(*[d[0] for d in data])]
    normalized_data = [
        [(x - min_val) / (max_val - min_val) if max_val > min_val else 0 for x, min_val, max_val in zip(row[0], min_values, max_values)]
        for row in data
    ]
    return [(norm, row[1]) for norm, row in zip(normalized_data, data)]

# Read and process the dataset
with open("adult.data", "r") as f:
    training_data = [parse_line(line, encoders) for line in f.readlines() if len(line) > 4]

normalized_training_data = normalize(training_data)
train, test = train_test_split(normalized_training_data, test_size=0.2)


class NeuralNet:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # Initialize the neural network
        pass
    
    def train(self, data, iterations, learning_rate, print_interval):
        # Train the neural network
        pass
