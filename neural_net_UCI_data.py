from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

from typing import Tuple, List 

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits a line of CSV into inputs and output, transforming the output as appropriate.

    Args:
        line: One line of the CSV as a string.

    Returns:
        A tuple of input list and output list.
    """
    # Split the line by comma, stripping spaces
    tokens = [token.strip() for token in line.split(',')]

    # Process the output (income), where '>50K' becomes [1.0] and '<=50K' becomes [0.0]
    output = [1.0 if tokens[-1] == '>50K' else 0.0]

    # Process inputs: skipping the income value and converting relevant fields to floats
    # Convert categorical variables to numerical codes if necessary (not done here)
    inputs = [float(tokens[i]) if i in {0, 2, 4, 10, 11, 12} else tokens[i] for i in range(len(tokens) - 1)]

    return (inputs, output)
def normalize(data: List[Tuple[List[float], List[float]]]):


    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("wine_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(13, 3, 1)
nn.train(train, iters=10000, print_interval=1000, learning_rate=0.2)

for i in nn.test_with_expected(test):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")
