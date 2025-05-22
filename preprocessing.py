import pandas as pd
import os
import numpy as np
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


    manufacturer = []
    model = []
    theoretical_seats = []

    for aircraft in df['airplane'].unique():
        print(f"Processing {aircraft} identification")
        code, country, alliance = aux_functions.standardize_aircraft(name = aircraft, choices = choices, aircrafts = aircrafts)
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

    vector_names = []
    vector_IATA = []

    for airline in df['airline'].unique():
        print(f"Processing {airline} identification")
        tmp_name, tmp_IATA = aux_functions.get_airline_code(airline,airline_to_iata)
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
    
    # Apply the mapping
    df['from'] = df['from'].map(aux_functions.code_mapping).fillna(df['from'])
    df['to'] = df['to'].map(aux_functions.code_mapping).fillna(df['to'])

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

    df['haversine'] = aux_functions.haversine(
        df['Latitude'],df['Longitude'],
        df['Latitude_to'],df['Longitude_to']
        )

    df['flight_time'] = aux_functions.estimate_flight_time(df['haversine'])

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
        
    subset_takeoff_reg_df['ids'] = subset_takeoff_reg_df.apply(
        lambda row: aux_functions.get_id(subset_landing, row), axis=1
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
    
    joint_lt_subset = joint_lt_subset.groupby(['route','IATA'], group_keys=False).apply(aux_functions.replace_minor_models)
    
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

    

