import pandas as pd
import re
import csv
import requests
import os
import json

def retrieve_csv():
    
    with open("config.json") as f:
        config = json.load(f)

    with open(config['csv_names'], "r") as file:
        links = [line.strip() for line in file]

    for link in links:
        
        year = re.search(r'/download/(\d{4})', link).group(1)

        if os.path.exists(f"data/flightdata_raw/flightdata_{year}.csv"):
            print(f"csv for year {year} already exists. Skipping download")
            continue
        
        response = requests.get(link)
        response.raise_for_status()
        content = response.text

        # Handling different csv delimiters
        sniffer = csv.Sniffer().sniff(content, delimiters=[',',';'])
        delimiter = sniffer.delimiter

        df = pd.read_csv(link, delimiter = delimiter, low_memory=False)

        new_filename = f"data/flightdata_raw/flightdata_{year}.csv"
        df.to_csv(new_filename)

        print(f"successfully saved flight data for year {year}")

    return 0