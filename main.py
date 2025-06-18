# src/main.py

import import_requests
import download_csv
import preprocessing
# import scrap_wiki
import download_csv_airports_info
import sys

def main():
    
    args = sys.argv[1:]

    print("Retrieveing csv links")

    import_requests.get_csv_links()

    print("Downloading csvs")

    download_csv.retrieve_csv()

    print("Scrap Airports info")
    
    download_csv_airports_info.get_airports_info()

    # scrap_wiki.get_airlines_info()

    print("Preprocessing data")

    preprocessing.preprocess_first_step()
    
    preprocessing.preprocess_first_step_b()
    
    preprocessing.preprocess_second_step(args)
    
    preprocessing.preprocess_third_step()

if __name__ == "__main__":

    main()