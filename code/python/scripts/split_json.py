import json
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(json_file_path, test_size):
    """
    Split a JSON dataset into train and test sets.

    Parameters:
    - json_file_path: Path to the JSON file.
    - test_size: Proportion of the dataset to include in the test split.

    Returns:
    - train_set: Training set.
    - test_set: Test set.
    """
    # Load the data from the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Split the data into train and test sets
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=42)

    return train_set, test_set

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Split a JSON dataset into train and test sets.")

    # Add the arguments
    parser.add_argument('--json_file', type=str, required=True, help="Path to the JSON file.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")

    # Execute the parse_args() method
    args = parser.parse_args()

    # Split the dataset
    train_set, test_set = split_dataset(args.json_file, args.test_size)

    # Optionally, save the split datasets back to JSON files
    with open('train_set.json', 'w') as file:
        json.dump(train_set, file)

    with open('test_set.json', 'w') as file:
        json.dump(test_set, file)

if __name__ == "__main__":
    main()

