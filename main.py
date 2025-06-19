# src/main.py

import logging
import import_requests
import download_csv
import preprocessing
import download_csv_airports_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    
    logger.info("Retrieving CSV links")

    import_requests.get_csv_links()

    logger.info("Downloading CSVs")

    download_csv.retrieve_csv()

    logger.info("Scraping airport info")
    
    download_csv_airports_info.get_airports_info()

    logger.info("Preprocessing data")

    preprocessing.preprocess_step_a()
    
    preprocessing.preprocess_step_b()
    
    preprocessing.preprocess_step_c()
    
    preprocessing.preprocess_step_d()

if __name__ == "__main__":

    main()