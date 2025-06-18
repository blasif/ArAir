import os
import pandas as pd

def get_airports_info():
    
    file_airports_name = 'aeropuertos_detalle.csv'

    if os.path.exists(file_airports_name):
        return 0
    else:
        airport_data = pd.read_csv('https://datos.transporte.gob.ar/dataset/62b3fe5f-ffe6-4d8f-9d59-bfabe75d1ee8/resource/eb54e49e-9a5a-4614-91f4-526c650d0105/download/aeropuertos_detalle.csv',sep=';')
        
        airport_data.to_csv(file_airports_name, index = False)

    return 0