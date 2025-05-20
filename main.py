# src/main.py

import import_requests
import download_csv
import preprocessing
import scrap_wiki

def main():

    print("Retrieveing csv links")

    import_requests.get_csv_links()

    print("Downloading csvs")

    download_csv.retrieve_csv()

    print("Scrap Airports info")

    scrap_wiki.get_airlines_info()

    print("Preprocessing data")

    preprocessing.preprocess_first_step()
    
    preprocessing.preprocess_first_step_b()
    
    preprocessing.preprocess_second_step()
    
    preprocessing.preprocess_third_step()

if __name__ == "__main__":

    main()