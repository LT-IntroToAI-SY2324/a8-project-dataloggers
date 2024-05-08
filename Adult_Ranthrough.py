from neural_net_UCI_data import parse_line
from neural_net_UCI_data import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from neural_net import NeuralNet  # Make sure NeuralNet is properly defined in neural_net.py

# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath, header=None, names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital-Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss",
        "Hours-per-Week", "Native-Country", "Income"
    ])
    
    # Handle missing values by replacing ' ?' with NaN and then dropping these rows
    data.replace(' ?', pd.NA, inplace=True)
    data.dropna(inplace=True)
    
    # Encode categorical variables using one-hot encoding
    categorical_cols = ["Workclass", "Education", "Marital-Status", "Occupation",
                        "Relationship", "Race", "Sex", "Native-Country"]
    data = pd.get_dummies(data, columns=categorical_cols)
    
    # Encode the target variable 'Income'
    data['Income'] = data['Income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    
    # Extract inputs and outputs
    inputs = data.drop('Income', axis=1).values
    outputs = data['Income'].values.reshape(-1, 1)
    
    return inputs, outputs

# Correct the filepath to point to your actual data file location
filepath = '/Users/isaacespadas/a8-project/adult dat set/adult.data'

inputs, outputs = load_and_preprocess_data(filepath)

# Split the data into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42, shuffle=True
)

# Initialize the neural network with appropriate dimensions
num_features = train_inputs.shape[1]  # Determine the number of input features dynamically
nn = NeuralNet(n_input=num_features, n_hidden=10, n_output=1)

# Train the neural network
nn.train(train_inputs, train_outputs, epochs=100, lr=0.1, momentum=0.1)

# Evaluate the model and calculate accuracy
# Assuming evaluate is now adjusted to handle multiple inputs:
predicted_outputs = nn.evaluate(test_inputs)  # Directly pass all test inputs if evaluate is adjusted
accuracy = (predicted_outputs.round() == test_outputs).mean()
print(f'Test Accuracy: {accuracy:.2%}')