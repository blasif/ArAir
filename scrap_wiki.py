import requests
from bs4 import BeautifulSoup
import re
import csv
import os
import pandas as pd

def get_airlines_info():

    airlines_csv_path = 'data/airline_codes.csv'

    if os.path.exists(airlines_csv_path):
        print(f"{airlines_csv_path} already exists. Skipping Airlines scrap.")
        return 0

    # Fetch the Wikipedia page
    url = "https://en.wikipedia.org/wiki/List_of_airline_codes"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with airline codes (usually the first table on the page)
    table = soup.find('table', {'class': 'wikitable'})

    # Extract table headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # Extract table rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header row
        row = [td.text.strip() for td in tr.find_all('td')]
        if row:  # Ignore empty rows
            rows.append(row)

    # Convert to DataFrame
    airlines = pd.DataFrame(rows, columns=headers)

    # Save to CSV
    airlines.to_csv(airlines_csv_path, index=False)

    return 0