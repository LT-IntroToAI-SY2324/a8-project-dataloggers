from neural_net_UCI_data import parse_line
from neural_net_UCI_data import normalize 

def process_and_print_dataset(filename: str):
    with open(filename, 'r') as file:
        for line in file:
            parsed_data = parse_line(line)
            normalized_data = normalize(parsed_data[0])  
            print(normalized_data, parsed_data[1])
            print(parsed_data)


dataset_path = "adult.data"
process_and_print_dataset(dataset_path)
