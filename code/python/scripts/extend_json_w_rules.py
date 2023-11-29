# Author: harpo Maxx
# Usage:
# python3 extend_json_w_rules.py --json_file ../../../rawdata/laboral/sumariosbigdb/sumariosdb.json \
#        --txt_directory ../../../rawdata/laboral/sumariosbigdb/resumenfallos/ 
#        --output sumarios.json

import argparse
import json
import os

def read_txt_file(directory, filename):
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."

def main(json_file, output_file, txt_directory):
    # Read JSON data from the file
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("JSON file not found.")
        return

    # Process each entry in the JSON data
    for entry in data:
        fallo = entry.get('fallo')
        if fallo:
            txt_filename = f"{fallo}_summary_claude.txt"
            sentencia = read_txt_file(txt_directory, txt_filename)
            entry['sentencia'] = sentencia

    # Write the modified JSON data to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("--json_file", help="Path to the JSON file")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    parser.add_argument("--txt_directory", help="Directory where .txt files are located")
    args = parser.parse_args()

    main(args.json_file, args.output_file, args.txt_directory)

