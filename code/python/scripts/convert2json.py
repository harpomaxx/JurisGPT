import csv
import json
import argparse

# Function to convert CSV to JSON
def convert_csv_to_json(csv_file_path, json_file_path):
    # Create a list to hold the rows as dictionaries
    data = []

    # Read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # Create a CSV reader
        csv_reader = csv.DictReader(file)

        # Loop over each row in the CSV
        for row in csv_reader:
            # Append the row (as a dictionary) to the data list
            data.append(row)

    # Write the data to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Convert a CSV file to a JSON file.')
    parser.add_argument('--input_csv', help='Input CSV file path')
    parser.add_argument('--output_json', help='Output JSON file path')

    # Parse the arguments
    args = parser.parse_args()

    # Perform the conversion
    convert_csv_to_json(args.input_csv, args.output_json)
    print(f"Converted {args.input_csv} to {args.output_json}")

if __name__ == "__main__":
    main()
