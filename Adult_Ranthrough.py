import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def initialize_encoders(data):
    encoders = {}
    categorical_indices = [1, 3, 5, 6, 7, 8, 9, 13]  # Assuming these are the indices of categorical columns
    for index in categorical_indices:
        encoder = LabelEncoder()
        data.iloc[:, index] = data.iloc[:, index].astype(str)  # Ensure data is the proper type
        encoder.fit(data.iloc[:, index])
        encoders[index] = encoder
    return encoders

def parse_line(tokens, encoders):
    inputs = []
    for i, token in enumerate(tokens[:-1]):
        if i in encoders:
            inputs.append(float(encoders[i].transform([token])[0]))
        else:
            inputs.append(float(token))
    output = [1.0 if tokens[-1].strip() == '>50K' else 0.0]
    return (inputs, output)

def process_and_print_dataset(filename: str):
    data = pd.read_csv(filename, header=None)
    encoders = initialize_encoders(data)
    scaler = StandardScaler()

    # Parse each row in the dataframe
    all_data = [parse_line(row, encoders) for _, row in data.iterrows()]
    inputs = [d[0] for d in all_data]

    # Normalize inputs
    normalized_inputs = scaler.fit_transform(inputs)

    # Print normalized data and output
    for norm, (_, output) in zip(normalized_inputs, all_data):
        print(list(norm), output)

dataset_path = "adult.data"
process_and_print_dataset(dataset_path)
