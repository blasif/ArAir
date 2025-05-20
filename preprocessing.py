import pandas as pd
import os
import numpy as np
from fuzzywuzzy import process as process_one
import aux_functions

def preprocess_first_step():

    combined_csv_path = 'data/flightdata_combined.csv'

    if os.path.exists(combined_csv_path):
        print(f"{combined_csv_path} already exists. Skipping preprocessing.")
        return 0

    csv_files = [f for f in os.listdir('data/flightdata_raw/') if f.endswith('.csv')]

    for iters, files in enumerate(csv_files):
    
        temp_df = pd.read_csv(f"data/flightdata_raw/{files}",low_memory = False,index_col=0, delimiter= ',', quotechar = '"', skipinitialspace=True, header=0)
    
        if iters > 0:
            temp_df.columns  = df.columns

        if iters == 0:
            df = temp_df
            continue

        df = pd.concat([df, temp_df])

    # ----------–-------------------
    # Step 1 : Preprocess
    # ------------------------------

    # Simple filtering and preprocessing of data
    
    df.rename(columns = {'Clase de Vuelo (todos los vuelos)' : 'flight',
                     'Fecha UTC' : 'date',
                     'Hora UTC' : 'utc',
                     'Clasificación Vuelo' : 'type',
                     'Tipo de Movimiento' : 'mov',
                     'Aeropuerto' : 'from',
                     'Pasajeros' : 'pas_num',
                     'Origen / Destino' : 'to',
                     'Aerolinea Nombre' : 'airline',
                     'Aeronave' : 'airplane',
                     'Calidad dato' : 'info' }, inplace = True)

    df.query(' flight == "Regular" & type == "Doméstico" ',inplace = True)

    df = df[['date','utc','mov','from','to','airline','airplane','pas_num','PAX','info']]

    df.drop('info', axis = 1, inplace = True)

    counts = df['airline'].value_counts()
    Index_Airlines = counts[counts > 100].index

    df = df[df['airline'].isin(Index_Airlines)]

    df['PAX'] = pd.to_numeric(df['PAX'].astype(str).str.replace(',', '.', regex=False))

    Index_airlines = df.groupby('airline')['PAX'].sum().loc[lambda x: x > 6000].index

    df = df[df['airline'].isin(Index_airlines)]
    df = df[df['airline'] != '0']
    df = df[df['airline'] != 'LADE']

    # ----------–-------------------
    # Step 2 : Update Airplane names
    # ------------------------------
     
    airlines = pd.read_csv("airline_codes.csv")
    
    aircrafts = pd.read_csv('ACFTREF.txt', delimiter=',', header=0, dtype=str)

    aircrafts['MFR'] =  aircrafts['MFR'].astype(str).str.strip()
    aircrafts['MODEL'] =  aircrafts['MODEL'].astype(str).str.strip()
    aircrafts['NO-SEATS'] = aircrafts['NO-SEATS'].astype(int)
    aircrafts['NO-ENG'] = aircrafts['NO-ENG'].astype(int)
    aircrafts['SPEED'] = aircrafts['SPEED'].astype(int) * 1.60934 # Miles to km

    choices = aircrafts['MFR'] + " " + aircrafts['MODEL']

    def standardize_aircraft(name):
        # Fuzzy match if not found
        best_match = process.extractOne(name, choices = choices)
        if best_match and best_match[1] > 60:  # threshold for confidence
            return aircrafts['MFR'].iloc[best_match[2]], aircrafts['MODEL'].iloc[best_match[2]], aircrafts['NO-SEATS'].iloc[best_match[2]]
        return 'UNKNOWN','UNKNOWN',0

    from rapidfuzz import process, fuzz

    manufacturer = []
    model = []
    theoretical_seats = []

    for aircraft in df['airplane'].unique():
        print(f"Processing {aircraft} identification")
        code, country, alliance = standardize_aircraft(aircraft)
        manufacturer.append(code)
        model.append(country)
        theoretical_seats.append(alliance)

    iata_to_aircraft = dict(zip(df['airplane'].unique(),manufacturer))
    df.loc[:,'BRAND'] = df['airplane'].map(iata_to_aircraft)
    iata_to_aircraft = dict(zip(df['airplane'].unique(),model))
    df.loc[:,'MODEL'] = df['airplane'].map(iata_to_aircraft)
    iata_to_aircraft = dict(zip(df['airplane'].unique(),theoretical_seats))
    df.loc[:,'THEO_SEATS'] = df['airplane'].map(iata_to_aircraft)

    # ---------------------------------------------
    # Step 3 : Retrieve IATA callsigns for Airlines
    # ---------------------------------------------

    airline_to_iata = dict(zip(airlines['Airline'], airlines['IATA']))    

    def get_airline_code(airline_name):
        """Fuzzy-match airline name to IATA code."""
        match, score = process.extractOne(airline_name.upper(), airline_to_iata.keys())
        print(f'{match}')
        if score >= 60:
            return match, airline_to_iata[match]
        else: 
            return None, None

    from fuzzywuzzy import process

    vector_names = []
    vector_IATA = []

    for airline in df['airline'].unique():
        print(f"Processing {airline} identification")
        tmp_name, tmp_IATA = get_airline_code(airline)
        vector_names.append(tmp_name)
        vector_IATA.append(tmp_IATA)

    vector_IATA[4] = 'ALA'
    vector_IATA[6] = 'Aerea'
    # vector_names[4] = "Andes LA"

    result = [f"{name} ({iata})" for name, iata in zip(vector_names, vector_IATA)]

    iata_to_airline = dict(zip(df['airline'].unique(),vector_IATA))
    df['IATA'] = df['airline'].map(iata_to_airline)

    iata_to_airline = dict(zip(df['airline'].unique(),result))
    df['AIR_STD'] = df['airline'].map(iata_to_airline)

    df.loc[:,'date'] = pd.to_datetime(df.date,dayfirst=True)

    df['EMP_SEATS'] = df.groupby(['IATA','BRAND','MODEL'])['PAX'].transform(lambda x : int(x.quantile(0.99)))

    df = df.reset_index(drop=True)

    df.loc[df['PAX'] > df['EMP_SEATS'] ,'PAX'] = df['EMP_SEATS']
    df['OCU_THEO'] = df['PAX'] / df['THEO_SEATS']
    df['OCU_EMP'] = df['PAX'] / df['EMP_SEATS']
    
    # ---------------------------------------------
    # Step 4 : new stuff
    # ---------------------------------------------
    
    times = pd.to_datetime(df['utc'], format='%H:%M').dt.time

    dates_dt = pd.to_datetime(df['date'])
    times_dt = pd.to_datetime(df['utc'], format='%H:%M')
    df['combined_hour'] = dates_dt + (times_dt - times_dt.dt.normalize())
    df = df.drop(labels = ['date','utc'], axis = 1)
    
    df.to_csv(combined_csv_path)
    

def preprocess_first_step_b():
    
    combined_csv_path_ss = 'data/flightdata_combined_b.csv'
    combined_csv_path_fs = 'data/flightdata_combined.csv'

    if os.path.exists(combined_csv_path_ss):
        print(f"{combined_csv_path_ss} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{combined_csv_path_fs}",low_memory = False,index_col=0)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'])
    
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

    # Apply the mapping
    df['from'] = df['from'].map(code_mapping).fillna(df['from'])
    df['to'] = df['to'].map(code_mapping).fillna(df['to'])

    df.loc[lambda row: row['from'] == 'IGU','from'] = 'IGR'
    df.loc[lambda row: row['to'] == 'IGU','to'] = 'IGR'

    df.loc[lambda row: row['from'] == 'RTA','from'] = 'RCQ'
    df.loc[lambda row: row['to'] == 'RTA','to'] = 'RCQ'

    df.loc[lambda row: row['from'] == 'MDB','from'] = 'MDQ'
    df.loc[lambda row: row['to'] == 'MDB','to'] = 'MDQ'

    Airports_info = pd.read_csv("Airports_info.csv", index_col = 0)

    # df.drop(labels = 'index', axis = 1, inplace = True)

    df = df.set_index('from').join(how = 'left',  other= Airports_info.set_index('IATA'),rsuffix = '_test').reset_index().set_index('to').join(how = 'left',  other= Airports_info.set_index('IATA'),rsuffix = '_to' ).reset_index()

    df.drop(labels = 'Coordinates',axis=1,inplace=True)
    df.drop(labels = 'Coordinates_to',axis=1,inplace=True)

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

    df['haversine'] = haversine(df['Latitude'],df['Longitude'],df['Latitude_to'],df['Longitude_to'])

    def estimate_flight_time(distance_km):
    
        # Calculate time in hours then convert to minutes
        flight_time_hours = distance_km / 850
        return flight_time_hours * 60

    df['flight_time'] = estimate_flight_time(df['haversine'])

    df = df.dropna()

    df['route'] = np.where(
        df['mov'] == 'Despegue',
        # Sort 'from' and 'to' alphabetically for a unique route
        np.where(
            df['from'] < df['to'], 
            df['from'] + ' - ' + df['to'],  # Keep original order if 'from' < 'to'
            df['to'] + ' - ' + df['from']   # Flip order if 'to' < 'from'
        ),
        # Same logic for landing ('Aterrizaje')
        np.where(
            df['to'] < df['from'], 
            df['to'] + ' - ' + df['from'],  # Flip order if 'to' < 'from'
            df['from'] + ' - ' + df['to']   # Keep original order if 'from' < 'to'
        )
    )
    
    df.reset_index(inplace=True)    
    
    df.to_csv(combined_csv_path_ss)
    

def preprocess_second_step():
        
    combined_csv_path_ss = 'data/flightdata_combined_second.csv'
    
    combined_csv_path_fs = 'data/flightdata_combined_b.csv'

    if os.path.exists(combined_csv_path_ss):
        print(f"{combined_csv_path_ss} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{combined_csv_path_fs}",low_memory = False,index_col=0)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'])

    subset_takeoff_reg_df = df.query("`mov` == 'Despegue'").copy()
    subset_landing = df.query(' mov == "Aterrizaje" ').copy()

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
        
    subset_takeoff_reg_df['ids'] = subset_takeoff_reg_df.apply(
        lambda row: get_id(subset_landing, row), axis=1
        )
    
    ids_list = subset_takeoff_reg_df['ids'].tolist()
    
    subset_takeoff_reg_df.to_csv(combined_csv_path_ss)
    
    
    
def preprocess_third_step():
    
    combined_csv_path_ss = 'data/flightdata_combined_third.csv'
    
    combined_csv_path_fs = 'data/flightdata_combined_second.csv'

    if os.path.exists(combined_csv_path_ss):
        print(f"{combined_csv_path_ss} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{combined_csv_path_fs}",low_memory = False,index_col=0)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'], yearfirst = True)
        df_old = pd.read_csv('data/flightdata_combined_b.csv',low_memory = False,index_col= 0)
        df_old['combined_hour'] = pd.to_datetime(df_old['combined_hour'], yearfirst = True)
        subset_landing = df_old.query("`mov` == 'Aterrizaje'").copy()
    
    df.dropna(subset = 'ids',inplace=True)
    df.ids = df.ids.astype(int)

    joint_lt = df.set_index('ids').join(how = 'left', other = subset_landing
    .set_index('index'), rsuffix = '_landing')

    joint_lt_subset = joint_lt[['from','to','mov','pas_num','pas_num_landing','PAX','PAX_landing','BRAND','MODEL','THEO_SEATS','IATA','AIR_STD','EMP_SEATS','OCU_THEO','OCU_EMP','OCU_THEO_landing','OCU_EMP_landing','combined_hour','combined_hour_landing','haversine','flight_time','route','Latitude','Longitude','Latitude_landing','Longitude_landing']]
    
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

    joint_lt_subset = joint_lt_subset.groupby(['route','IATA'], group_keys=False).apply(replace_minor_models)
    
    joint_lt_subset['time_taken'] = (joint_lt_subset['combined_hour_landing'] - joint_lt_subset['combined_hour']).dt.total_seconds() / 60

    joint_lt_subset['mean_flight_time'] = (
    joint_lt_subset.groupby(['IATA', 'from', 'to', 'BRAND', 'MODEL'])[['time_taken']]
    .transform('median')
    )

    joint_lt_subset['time_delay'] = (joint_lt_subset['time_taken'] - joint_lt_subset['mean_flight_time']) / 60 

    joint_lt_subset['count_routes'] = joint_lt_subset.groupby('route')['route'].transform('count')

    joint_lt_subset = joint_lt_subset.loc[joint_lt_subset['count_routes'] > 5,:]
    
    joint_lt_subset.to_csv(combined_csv_path_ss)
    
    return 0

    

