from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

<<<<<<< HEAD
from typing import Tuple, List 

=======
>>>>>>> 090a2a88e3ce213c908d88208d08ce3a826fc25d
def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output, transforming categorical data to numerical and output to binary.

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.strip().split(", ")
    
    # Example mappings for categorical fields - adjust based on your dataset knowledge
    # Mapping dictionaries for categorical fields
    workclass_map = {
    '?': -1, 'Federal-gov': 0, 'Local-gov': 1, 'Never-worked': 2, 'Private': 3,
    'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6, 'Without-pay': 7
}

    education_map = {
    '10th': 0, '11th': 1, '12th': 2, '1st-4th': 3, '5th-6th': 4, '7th-8th': 5,
    '9th': 6, 'Assoc-acdm': 7, 'Assoc-voc': 8, 'Bachelors': 9, 'Doctorate': 10,
    'HS-grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-school': 14, 'Some-college': 15
}

    marital_status_map = {
    'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
    'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6
}

    occupation_map = {
    '?': -1, 'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3,
    'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7,
    'Priv-house-serv': 8, 'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11,
    'Tech-support': 12, 'Transport-moving': 13
}

    relationship_map = {
    'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3,
    'Unmarried': 4, 'Wife': 5
}

    race_map = {
    'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4
}

    sex_map = {
    'Female': 0, 'Male': 1
}

    country_map = {
    '?': -1, 'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4,
    'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8, 'France': 9,
    'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13, 'Holand-Netherlands': 14,
    'Honduras': 15, 'Hong': 16, 'Hungary': 17, 'India': 18, 'Iran': 19,
    'Ireland': 20, 'Italy': 21, 'Jamaica': 22, 'Japan': 23, 'Laos': 24,
    'Mexico': 25, 'Nicaragua': 26, 'Outlying-US(Guam-USVI-etc)': 27, 'Peru': 28,
    'Philippines': 29, 'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32, 'Scotland': 33,
    'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37, 'United-States': 38,
    'Vietnam': 39, 'Yugoslavia': 40
}





    # Convert categories to numbers using the maps
    age = float(tokens[0].strip())
    workclass = workclass_map.get(tokens[1].strip(), -1)
    fnlwgt = float(tokens[2].strip())
    education_num = float(tokens[4].strip())
    marital_status = marital_status_map.get(tokens[5].strip(), -1)
    occupation = occupation_map.get(tokens[6].strip(), -1)
    relationship = relationship_map.get(tokens[7].strip(), -1)
    race = race_map.get(tokens[8].strip(), -1)
    sex = sex_map.get(tokens[9].strip(), -1)
    capital_gain = float(tokens[10].strip())
    capital_loss = float(tokens[11].strip())
    hours_per_week = float(tokens[12].strip())
    country = country_map.get(tokens[13].strip(), -1)

    # Input features
    inputs = [age, workclass, fnlwgt, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, country]
    
    # Output classification
    output = [1.0 if tokens[14] == '>50K' else 0.0]

    return (inputs, output)



def normalize(data: List[Tuple[List[float], List[float]]]):
    """Normalize the numeric input features of the data to the range 0-1.

    Args:
        data: List of tuples, where each tuple consists of a list of inputs and a list of outputs.

    Returns:
        Normalized data where each numeric input feature is scaled to the range 0-1.
    """
    if not data:
        return data  # Return the empty list if data is empty

    # Identify indices of numeric and categorical features
    # Assume numeric indices as per your dataset description (age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week)
    numeric_indices = [0, 2, 3, 9, 10, 11]
    num_features = len(data[0][0])  # Number of features, assuming all inputs are the same length

    # Initialize min and max lists for numeric features
    mins = [float('inf')] * num_features
    maxs = [float('-inf')] * num_features

    # Find the min and max for each numeric feature
    for inputs, _ in data:
        for index in numeric_indices:
            if inputs[index] < mins[index]:
                mins[index] = inputs[index]
            if inputs[index] > maxs[index]:
                maxs[index] = inputs[index]

    # Normalize the data
    normalized_data = []
    for inputs, outputs in data:
        normalized_inputs = inputs[:]  # Create a copy of the inputs to modify
        for index in numeric_indices:
            if maxs[index] != mins[index]:  # Avoid division by zero
                normalized_inputs[index] = (inputs[index] - mins[index]) / (maxs[index] - mins[index])
            else:
                normalized_inputs[index] = 0  # If max equals min, set to 0 to avoid NaN
        normalized_data.append((normalized_inputs, outputs))

    return normalized_data

file_path ="/Users/isaacespadas/a8-project/adult dat set/adult.data"
with open(file_path, 'r') as file:
    lines = file.readlines()

    # Loop through each line in the file
    for line in lines:
        if line.strip():  # This check ensures that you are not processing empty lines
            parsed_data = parse_line(line)
            print(f"Original: {line.strip()}")
            print(f"Parsed: {parsed_data}")
            print("-------------------------------")  # Separator for readability
# with open(file_path, "r") as file:
#     training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
# td = normalize(training_data)
# print(td)

# train, test = train_test_split(td)

# nn = NeuralNet(13, 3, 1)
# nn.train(train, iters=10000, print_interval=1000, learning_rate=0.2)

# for i in nn.test_with_expected(test):
#     difference = round(abs(i[1][0] - i[2][0]), 3)
#     print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")
