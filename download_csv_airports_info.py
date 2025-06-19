import os
import pandas as pd
import json
def get_airports_info():
    
    with open('config.json') as f:
        config = json.load(f)

    if os.path.exists(config["path_airports"]):
        return 0
    else:
        
        airport_data = pd.read_csv(config['airport_data_url'], sep = ';')
        
        airport_data.to_csv(config["path_airports"], index = False)

    return 0