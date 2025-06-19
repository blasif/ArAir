import requests
from bs4 import BeautifulSoup
import re
import os
import json

def get_csv_links():
    
    with open('config.json') as f:
        config = json.load(f)

    csv_file_path = config['csv_names'] 

    if os.path.exists(csv_file_path):
        print(f"{csv_file_path} already exists. Skipping link extraction.")
        return 0
    
    try:
        response = requests.get(config['flights_data_url'])
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    csv_links = []

    # Find all <a> tags that contain .csv links
    for a_tag in soup.find_all('a', href=re.compile(r'\.csv$')):
        csv_links.append(a_tag['href'])

    with open(csv_file_path, "w") as file:
        for item in csv_links:
            file.write(f"{item}\n")

    return 0