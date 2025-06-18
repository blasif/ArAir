import pandas as pd
from fuzzywuzzy import process as process_one
from rapidfuzz import process as process_two
import numpy as np
import unicodedata

def extract_brand(name):
    if pd.isna(name) or name == "0":  
        return None
    return name.split('-')[0]  

def extract_airplane_type(name):
    if pd.isna(name) or name == "0":  
        return None
    return name.split('-')[1]  

def extract_airplane_config(name):
    if pd.isna(name) or name == "0":  
        return None
    
    to_return = name.split('-')
    if(len(to_return) < 3):
        return None
    
    return to_return[2] 

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    
    # Radius of earth in kilometers (use 3956 for miles)
    r = 6371
    return c * r

def estimate_flight_time(distance_km):
    
    # Calculate time in hours then convert to minutes
    flight_time_hours = distance_km / 850
    return flight_time_hours * 60
    
def time_of_day_seconds(dt):
    """Convert datetime to seconds since midnight."""
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def circular_mean(times_in_seconds):
    """Compute the circular mean of times given in seconds."""
    radians = times_in_seconds * 2 * np.pi / 86400  # 86400 seconds in a day
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    mean_seconds = mean_angle * 86400 / (2 * np.pi)
    return mean_seconds

def centroid_departure_time(group):
    times_in_seconds = group['combined_hour'].dt.time.apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )
    mean_seconds = circular_mean(times_in_seconds)
    # Convert back to time
    hours = int(mean_seconds // 3600)
    minutes = int((mean_seconds % 3600) // 60)
    seconds = int(mean_seconds % 60)
    return pd.Timestamp(f"{hours:02d}:{minutes:02d}:{seconds:02d}").time()

def standardize_aircraft(name, choices, aircrafts):
    
    if name == "0":
        return 'UNKNOWN','UNKNOWN'
        
    best_match = process_two.extractOne(name, choices = choices)
    
    if best_match and best_match[1] > 60:  
        return aircrafts['MFR'].iloc[best_match[2]], aircrafts['MODEL'].iloc[best_match[2]]
    else:
        return 'UNKNOWN','UNKNOWN'
    
def get_id(df,row):
        
    mask = (
            (abs(df['combined_hour'] - ( row['combined_hour'] + pd.Timedelta(row['flight_time'],unit = 'm'))) < pd.Timedelta('2h')) & 
            (df['route'] == row['route']) &
            (df['index'] > row['index']) 
        )
        
    matches = df.loc[mask, 'index']
        
    if len(matches) > 0:
        return matches.iloc[0]
    else:
        return None
    
    
def replace_minor_models(group):
    # Count each model
    counts = group['MODEL'].value_counts(normalize=True)
    # Find the most common model and its proportion
    top_model = counts.idxmax()

    top_brand_idx = group['BRAND'].value_counts(normalize=True)
    top_brand = top_brand_idx.idxmax()

    theo_seats_idx = group['THEO_SEATS'].value_counts(normalize=True)
    top_seats = theo_seats_idx.idxmax()
    
    top_OCU_idx = group['OCU_THEO_landing'].value_counts(normalize=True)
    top_OCU = top_OCU_idx.idxmax()

    top_prop = counts.max()
        
    # If the top model is used >90% of the time, replace others
    if top_prop > 0.9:
        group['MODEL'] = top_model
        group['BRAND'] = top_brand
        group['THEO_SEATS'] = top_seats
        group['OCU_THEO_landing'] = top_OCU

    return group

def remove_accents(text):
    # Normalize the string to NFKD Unicode form
    text_nfkd = unicodedata.normalize('NFKD', str(text))
    # Filter out diacritical marks (accents)
    return ''.join(c for c in text_nfkd if not unicodedata.combining(c))