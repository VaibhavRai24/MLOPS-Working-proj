import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
from src.logger import logging


def load_params(params_path:str) ->dict:
    """
    Load parameters from the params.yaml file.
    
    Args:
    params_path : str : path to the params.yaml file
    
    Returns:
    dict : parameters from the params.yaml file
    """
    
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters loaded successfully.")
        return params
    except FileNotFoundError:
        logging.error("FileNotFoundError: The file was not found.")
        raise 
    except yaml.YAMLError as e:
        logging.error(f"YAMLError: {e}")
        raise
    except Exception as e:
        logging.error("Unexpected error has taken place:")
        raise
    
def load_data(data_url:str) ->pd.DataFrame:
    """
    Load data from the specified URL.
    
    """
    
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded successfully.")
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the recived file")
        raise 
    except Exception as e:
        logging.error("Unexpected error has taken place:")
        raise    
    
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data 
    
    """
    try:
        logging.info("Preprocessing the data has been started------------------------------------------------------------>.")
        final_dataframe = df[df["sentiment"].isin(["positive", "negative"])]
        final_dataframe["sentiment"] = final_dataframe["sentiment"].map({"positive":1, "negative":0})
        logging.info("Preprocessing the data has been completed----------------------------------------------------------->.")
        return final_dataframe
    except KeyError as e:
        logging.error("Missing columns in the dataframe.")
        raise 
    except Exception as e:
        logging.error("Unexpected error has taken place:")
        raise
    
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str) ->None:
    """save the data to the specified path"""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train_data.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test_data.csv"), index=False)
        logging.info("Data saved successfully.")
    except Exception as e:
        logging.error("Unexpected error has taken place while saving the data:")
        raise        
    
def main():
    try:
        params = load_params(params_path= "params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        # test_size = 0.2
        
        df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        
        final_dataframe = preprocess_data(df)
        train_data, test_data = train_test_split(final_dataframe, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path= './data')
        
    except Exception as e:
        logging.error("Unexpected error has taken place while running the data ingestion process:")
        print(f'Error: {e}')
        
        
if __name__ == "__main__":
    main()
