import pandas as pd
import os
import numpy as np
import aux_functions
import json

def preprocess_step_a():
    
    with open("config.json") as f:
        config = json.load(f)

    flight_data = config["path_flight_step_a"]
    path_flightdata_raw = config["path_flight_data_raw"]

    if os.path.exists(flight_data):
        print(f"{flight_data} already exists. Skipping preprocessing.")
        return 0

    csv_files = [f for f in os.listdir(path_flightdata_raw) if f.endswith('.csv')]

    for iters, files in enumerate(csv_files):
    
        temp_df = pd.read_csv(f"{path_flightdata_raw}{files}",low_memory = False,index_col=0, delimiter= ',', quotechar = '"', skipinitialspace=True, header=0)
    
        if iters > 0:
            temp_df.columns  = df.columns
        else:
            df = temp_df
            continue

        df = pd.concat([df, temp_df])

    # ----------–-------------------
    # Step 1 : Preprocess
    # ------------------------------
    
    df['Tipo de Movimiento'] = df['Tipo de Movimiento'].replace({'Aterrizaje' : 'landing', 'Despegue' : 'takeoff'})
    
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
    df.drop(['flight','type','info'], axis=1,  inplace = True)
    
    Index_Airlines = df['airline'].value_counts().sort_values(ascending=False)[0:6].index

    df = df[df['airline'].isin(Index_Airlines)].copy()

    df['PAX'] = pd.to_numeric(df['PAX'].astype(str).str.replace(',', '.', regex=False))
    
    df['date'] = pd.to_datetime(df['date'],dayfirst=True)
    df['utc'] = pd.to_datetime(df['utc'], format='%H:%M')
    df['from'] = df['from'].astype("string")
    df['to'] = df['to'].astype("string")
    df['airline'] = df['airline'].astype("string")
    df['airplane'] = df['airplane'].astype("string")
    df['PAX'] = df['PAX'].astype("int")
    df['combined_hour'] = df['date'] + (df['utc'] - df['utc'].dt.normalize())
    df.drop(['date','utc'],axis= 1 , inplace=True)
    
    # ----------–-------------------
    # Step 2 : Update Airplane names
    # ------------------------------
    
    path_aircraft_data = config["path_aircraft_data"]
        
    aircrafts = pd.read_csv(path_aircraft_data, delimiter = ',', header = 0, dtype = str)

    aircrafts['MFR'] =  aircrafts['MFR'].astype(str).str.strip()
    aircrafts['MODEL'] =  aircrafts['MODEL'].astype(str).str.strip()

    choices = aircrafts['MFR'] + " " + aircrafts['MODEL']

    manufacturer = []
    model = []

    for aircraft in df['airplane'].unique():
        print(f"Processing {aircraft} identification")
        manuf, mod = aux_functions.standardize_aircraft(name = aircraft, choices = choices, aircrafts = aircrafts)
        manufacturer.append(manuf)
        model.append(mod)

    iata_to_aircraft = dict(zip(df['airplane'].unique(),manufacturer))
    df.loc[:,'BRAND'] = df['airplane'].map(iata_to_aircraft)
    iata_to_aircraft = dict(zip(df['airplane'].unique(),model))
    df.loc[:,'MODEL'] = df['airplane'].map(iata_to_aircraft)
    
    # Impute missing airplane names
    
    df.reset_index(inplace=True)
    index_missing = df.query(' airplane == "0" ').index
    
    # 1. Get all relevant groups for the missing rows
    missing_info = df.loc[index_missing, ['from', 'to', 'airline']].drop_duplicates()

    # 2. For each group of ('from', 'to', 'airline'), process once
    for _, row in missing_info.iterrows():
        from_val, to_val, airline_val = row['from'], row['to'], row['airline']

        # Subset for this route+airline group
        subset = df[
            (df['from'] == from_val) &
            (df['to'] == to_val) &
            (df['airline'] == airline_val)
        ]

        if subset.empty:
            continue

        # Brand dominance condition
        MNF_sizes = subset.groupby('BRAND').size()
        dominant_ratio = (MNF_sizes / subset.shape[0]).max()

        if dominant_ratio > 0.9:
            brand_to_impute = MNF_sizes.idxmax()
            model_to_impute = (
                subset[subset['BRAND'] == brand_to_impute]
                .groupby('MODEL')
                .size()
                .idxmax()
            )

            # Apply the imputation to all matching missing rows
            mask = (
                (df['from'] == from_val) &
                (df['to'] == to_val) &
                (df['airline'] == airline_val) &
                (df.index.isin(index_missing))
            )
            df.loc[mask, ['BRAND', 'MODEL']] = brand_to_impute, model_to_impute
            
    df['airplane_std'] = df['BRAND'] + " " + df['MODEL']

    # ---------------------------------------------
    # Step 3 : Retrieve IATA callsigns for Airlines
    # ---------------------------------------------

    unique_airline_names = df['airline'].unique()

    map_airlines_names = dict({'A-Jet Aviation Aircraft Management': 'A-Jet AAM',
                            'Andes Líneas Aéreas' : 'Andes LA',
                            'AEROLINEAS ARGENTINAS SA' : 'Aerolineas Argentinas',
                            'AUSTRAL LINEAS AEREAS-CIELOS DEL SUR S.A' : 'Austral LA',
                            'FB LÍNEAS AÉREAS - FLYBONDI' : 'Flybondi',
                            'LADE' : 'LADE',
                            'JETSMART AIRLINES S.A.' : 'Jetsmart',
                            'LAN ARGENTINA S.A. (LATAM AIRLINES)' : 'LAN Argentina'})

    map_iata_names = dict({'A-Jet Aviation Aircraft Management': 'JAS',
                            'Andes Líneas Aéreas' : 'O4',
                            'AEROLINEAS ARGENTINAS SA' : 'AR',
                            'AUSTRAL LINEAS AEREAS-CIELOS DEL SUR S.A' : 'AU',
                            'FB LÍNEAS AÉREAS - FLYBONDI' : 'FO',
                            'LADE' : 'LADE',
                            'JETSMART AIRLINES S.A.' : 'JA',
                            'LAN ARGENTINA S.A. (LATAM AIRLINES)' : 'LA'})

    df['IATA'] = df['airline'].map(map_iata_names)
    df['Airline_polished'] = df['airline'].map(map_airlines_names)
    
    df['AIRLINE_STD'] = df['Airline_polished'] + " (" + df['IATA'] + ")"
    
    #std_names = [airline + " " + "(" + iata + ")" for airline, iata in zip(map_airlines_names.values(), map_iata_names.values())]

    #result = dict(zip(list(unique_airline_names),std_names))

    #df['AIRLINE_STD'] = df['airline'].map(result)

    df['EMP_SEATS'] = df.groupby(['IATA','BRAND','MODEL'])['PAX'].transform(lambda x : int(x.quantile(0.99)))

    df.loc[df['PAX'] > df['EMP_SEATS'] ,'PAX'] = df['EMP_SEATS']
    
    # df.drop('airline', axis = 1,inplace=True)
    
    df.to_csv(flight_data)
    

def preprocess_step_b():
    
    with open("config.json") as f:
        config = json.load(f)
    
    path_step_b = config['path_flight_step_b']
    path_step_a = config['path_flight_step_a']

    if os.path.exists(path_step_b):
        print(f"{path_step_b} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{path_step_a}",low_memory = False,index_col=0).drop('index', axis=1)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'])
        
    Airports_info = pd.read_csv(config['path_airports'], delimiter = ';')
        
    Airports_info['denominacion'] = [aux_functions.remove_accents(denominacion) for denominacion in Airports_info['denominacion']]

    Airports_info['ref'] = [aux_functions.remove_accents(ref) for ref in Airports_info['ref']]

    Airports_info['provincia'] = [aux_functions.remove_accents(provincia) for provincia in Airports_info['provincia']]

    Airports_info = Airports_info[['local','latitud','longitud','ref','distancia_ref','fir','provincia']].copy().rename(columns = {'latitud' : 'Latitude', 'longitud' : 'Longitude','ref' : 'City', 'distancia_ref' : 'dist_to_city', 'provincia' : 'province'})
    
    df = df.set_index('from').join(how = 'left',  other= Airports_info.set_index('local')).reset_index().set_index('to').join(how = 'left',  other= Airports_info.set_index('local'),rsuffix = '_to' ).reset_index()

    df.dropna(inplace = True)
    
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

    df.to_csv(path_step_b)
    
    
def preprocess_step_c():
    
    with open("config.json") as f:
        config = json.load(f)
    
    path_step_c = config['path_flight_step_c']
    path_step_b = config['path_flight_step_b']
        
    if os.path.exists(path_step_c):
        print(f"{path_step_c} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{path_step_b}",low_memory = False,index_col=0)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'])

    subset_takeoff_reg_df = df.query("`mov` == 'takeoff'").copy()
    subset_landing = df.query(' mov == "landing" ').copy()
        
    subset_takeoff_reg_df['ids'] = subset_takeoff_reg_df.apply(
        lambda row: aux_functions.get_id(subset_landing, row), axis=1
        )
        
    subset_takeoff_reg_df.to_csv(path_step_c)
    
def preprocess_step_d():
    
    # drop _to_landing
    
    with open("config.json") as f:
        config = json.load(f)
    
    path_step_d = config['path_flight_step_d']
    path_step_c = config['path_flight_step_c']
    path_step_b = config['path_flight_step_b']  
        
    if os.path.exists(path_step_d):
        print(f"{path_step_d} already exists. Skipping preprocessing.")
        return 0
    else :
        df = pd.read_csv(f"{path_step_c}",low_memory = False,index_col=0)
        df['combined_hour'] = pd.to_datetime(df['combined_hour'], yearfirst = True)
        df_old = pd.read_csv(path_step_b,low_memory = False,index_col= 0)
        df_old['combined_hour'] = pd.to_datetime(df_old['combined_hour'], yearfirst = True)
        subset_landing = df_old.query(" `mov` == 'landing'").copy()
    
    df.dropna(subset = 'ids', inplace = True)
    df.ids = df.ids.astype(int)

    joint_lt = df.set_index('ids').join(how = 'left', other = subset_landing
    .set_index('index'), rsuffix = '_landing')

    joint_lt_subset = joint_lt.drop(['index','mov','airline','airplane','to_landing','from_landing','mov_landing','airline_landing','airplane_landing','BRAND_landing','MODEL_landing','airplane_std_landing','IATA_landing','Airline_polished_landing','AIRLINE_STD_landing','EMP_SEATS_landing','Latitude_to_landing','Longitude_to_landing','City_to_landing','dist_to_city_to_landing','fir_to_landing','province_to_landing','haversine_landing','flight_time_landing','route_landing'], axis = 1)
    
    joint_lt_subset['OCU_EMP'] = joint_lt_subset['PAX'] / joint_lt_subset['EMP_SEATS']
    
    joint_lt_subset = joint_lt_subset.drop(joint_lt_subset[joint_lt_subset['from'] == joint_lt_subset['to']].index,axis = 0)
    joint_lt_subset = joint_lt_subset.drop(joint_lt_subset.query(' `OCU_EMP` == 0 ').index,axis=0)
    
    joint_lt_subset.to_csv(path_step_d)
    
    return 0

    

