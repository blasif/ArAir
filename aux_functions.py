import requests
from bs4 import BeautifulSoup
import re
import csv
import os
import pandas as pd
from fuzzywuzzy import process

def extract_brand(name):
    """Extract the brand of the airplane (text before the first dash)."""
    if pd.isna(name) or name == "0":  # Handle missing or invalid values
        return None
    return name.split('-')[0]  # Split by dash and return the first part

def extract_airplane_type(name):
    """Extract the brand of the airplane (text before the first dash)."""
    if pd.isna(name) or name == "0":  # Handle missing or invalid values
        return None
    return name.split('-')[1]  # Split by dash and return the first part

def extract_airplane_config(name):
    """Extract the brand of the airplane (text before the first dash)."""
    if pd.isna(name) or name == "0":  # Handle missing or invalid values
        return None
    
    to_return = name.split('-')
    if(len(to_return) < 3):
        return None
    
    return to_return[2]  # Split by dash and return the first part

def get_id(df,row):
    # First check if all required columns exist
    required_columns = ['IATA','from', 'to', 'mov', 'pas_num', 'airplane']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise KeyError(f"DataFrame missing required columns: {missing_cols}")
    
    try:
        mask = (
            (abs(df['combined_hour'] - row['combined_hour']) < pd.Timedelta('10h')) & 
            (df['IATA'] == row['IATA']) &
            (df['from'] == row['to']) & 
            (df['to'] == row['from']) &
            (df['mov'] == 'Aterrizaje') &
            (abs(df['pas_num'] - row['pas_num']) <= 5) &
            (abs(df['PAX'] - row['PAX']) <= 5) &
            (df['index'] > row['index']) 
        )
        
        matches = df.loc[mask, 'index']
        
        if len(matches) == 1:
            return matches.iloc[0] if not matches.empty else None
        if len(matches) > 1:
            return matches.iloc[0]
        return None  # or handle multiple/no matches as needed
        
    except KeyError as e:
        print(f"Missing key in row data: {e}")
        return None

def get_airline_code(airline_name, airline_to_iata):
    """Fuzzy-match airline name to IATA code."""
    match, score = process.extractOne(airline_name.upper(), airline_to_iata.keys())
    return airline_to_iata[match] if score >= 80 else None

# Mapping dictionary for non-standard codes
code_mapping = {
    'ARR': 'ARR',
    'IGU': 'IGU',
    'AER': 'AEP',  # Aeroparque
    'BAR': 'BRC',  # Bariloche
    'GRA': 'RGL',  # Rio Gallegos
    'ECA': 'FTE',  # El Calafate (check this)
    'NEU': 'NQN',  # Neuquén
    'CBA': 'COR',  # Córdoba
    'USU': 'USH',  # Ushuaia
    'CRR': 'CRD',  # Comodoro Rivadavia
    'POS': 'PSS',  # Posadas
    'MDP': 'MDQ',  # Mar del Plata
    'SAL': 'SLA',  # Salta
    'DOZ': 'MDZ',  # Mendoza (possibly)
    'CHP': 'CPC',  # San Martin de los Andes (Chapelco)
    'JUA': 'JUJ',  # Jujuy (duplicate)
    'BCA': 'BHI',  # Bahia Blanca (check)
    'PAR': 'PRA',  # Paraná
    'ESQ': 'EQS',  # Esquel
    'TRE': 'REL',  # Trelew
    'OSA': 'OES',  # San Antonio Oeste
    'VIE': 'VDM',  # Viedma
    'SVO': 'SVI',  # Not in Argentina (Russian airport)
    'FSA': 'ROS',  # Rosario (Fisherton)
    'SIS': 'SGV',  # Sierra Grande
    'TRH': 'TDL',  # Tandil
    'CRV': 'CRD',  # Comodoro Rivadavia (duplicate)
    'DRY': 'RYO',  # Rio Turbio
    'GAL': 'IGR',  # Iguazu (check) !!!!!!!!!!!!!!
    'SDE': 'SDE',  # Correct (Santiago del Estero)
    'LAR': 'IRJ',  # La Rioja
    'SRA': 'RSA',  # Santa Rosa
    'CAT': 'CTC',  # Catamarca
    'UIS': 'UAQ',  # San Juan
    'TRC': 'TUC'   # Tucumán (duplicate)
}

def haversine(lat1, lon1, lat2, lon2):
    import numpy as np
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


def get_id(df,row):
        
    mask = (
            (abs(df['combined_hour'] - ( row['combined_hour'] + pd.Timedelta(row['flight_time'],unit = 'm'))) < pd.Timedelta('3h')) & 
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
    from rapidfuzz import process, fuzz
    # Fuzzy match if not found
    best_match = process.extractOne(name, choices = choices)
    if best_match and best_match[1] > 60:  # threshold for confidence
        return aircrafts['MFR'].iloc[best_match[2]], aircrafts['MODEL'].iloc[best_match[2]], aircrafts['NO-SEATS'].iloc[best_match[2]]
    return 'UNKNOWN','UNKNOWN',0


def get_airline_code(airline_name, airline_to_iata):
    from fuzzywuzzy import process
    """Fuzzy-match airline name to IATA code."""
    match, score = process.extractOne(airline_name.upper(), airline_to_iata.keys())
    print(f'{match}')
    if score >= 60:
        return match, airline_to_iata[match]
    else: 
        return None, None
    

# Mapping dictionary for non-standard codes
code_mapping = {
    'ARR': 'ARR',
    'IGU': 'IGU',
    'AER': 'AEP',  # Aeroparque
    'BAR': 'BRC',  # Bariloche
    'GRA': 'RGL',  # Rio Gallegos
    'ECA': 'FTE',  # El Calafate (check this)
    'NEU': 'NQN',  # Neuquén
    'CBA': 'COR',  # Córdoba
    'USU': 'USH',  # Ushuaia
    'CRR': 'CRD',  # Comodoro Rivadavia
    'POS': 'PSS',  # Posadas
    'MDP': 'MDQ',  # Mar del Plata
    'SAL': 'SLA',  # Salta
    'DOZ': 'MDZ',  # Mendoza (possibly)
    'CHP': 'CPC',  # San Martin de los Andes (Chapelco)
    'JUA': 'JUJ',  # Jujuy (duplicate)
    'BCA': 'BHI',  # Bahia Blanca (check)
    'PAR': 'PRA',  # Paraná
    'ESQ': 'EQS',  # Esquel
    'TRE': 'REL',  # Trelew
    'OSA': 'OES',  # San Antonio Oeste
    'VIE': 'VDM',  # Viedma
    'SVO': 'SVI',  # Not in Argentina (Russian airport)
    'FSA': 'ROS',  # Rosario (Fisherton)
    'SIS': 'SGV',  # Sierra Grande
    'TRH': 'TDL',  # Tandil
    'CRV': 'CRD',  # Comodoro Rivadavia (duplicate)
    'DRY': 'RYO',  # Rio Turbio
    'GAL': 'IGR',  # Iguazu (check) !!!!!!!!!!!!!!
    'SDE': 'SDE',  # Correct (Santiago del Estero)
    'LAR': 'IRJ',  # La Rioja
    'SRA': 'RSA',  # Santa Rosa
    'CAT': 'CTC',  # Catamarca
    'UIS': 'UAQ',  # San Juan
    'TRC': 'TUC'   # Tucumán (duplicate)
}


def get_id(df,row):
        
    mask = (
            (abs(df['combined_hour'] - ( row['combined_hour'] + pd.Timedelta(row['flight_time'],unit = 'm'))) < pd.Timedelta('3h')) & 
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