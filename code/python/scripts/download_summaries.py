# Script for donwloading summaries from the web
# 
import csv
import os
import requests
from html2text import HTML2Text

# Set directory to save files in
save_dir = "output/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open CSV file for reading
with open("laboral_sheet2.csv") as csv_file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csv_file)

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract 'fallo' value and add .txt to it
        file_name = row["fallo"] + ".txt"
        # Create URL from 'web' value
        url = row["web"]
        try:
            # Send GET request to URL and store response in variable
            response = requests.get(url)
            # Convert HTML to plain text using html2text library
            content = HTML2Text().handle(response.content.decode())
            # Save plain text as ASCII file in directory with given filename
            with open(os.path.join(save_dir, file_name), "w") as output_file:
                output_file.write(content)
        except Exception as e:
            print("Error:", e)
