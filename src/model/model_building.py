import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging


def load_data(file_path:str) ->pd.DataFrame:
    """
    This basically load the data from the given path
    
    """
    try:
        df =  pd.read_csv(file_path)
        logging.info('Data has been loaded successfully from the given path')
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the csv file %s', file_path)
        raise
    except FileNotFoundError:
        logging.error('File not found at the given path %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected thing has taken place, have a look in load_data', exc_info= True)
        raise e
    
    
def train_the_model(X_train:np.ndarray, y_train:np.ndarray) ->LogisticRegression:
    """
    This function is used to train the model
    """
    try:
        logging.info('Training the model........................')
        Model = LogisticRegression(C=1.0, solver= 'liblinear', random_state= 42, penalty= 'l1')
        Model.fit(X_train, y_train)
        logging.info('Model has been trained successfully')
        return Model
    except Exception as e:
        logging.error('Unexpected thing has taken place, have a look in train_the_model', exc_info= True)
        raise e
    
def save_model(model, file_path:str) ->None:
    """
    This function is used to save the model
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model has been saved successfully')
    except Exception as e:
        logging.error('Unexpected thing has taken place, have a look in save_model', exc_info= True)
        raise e
    

def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        model = train_the_model(X_train, y_train)
        save_model(model, './models/model.pkl')
    except Exception as e:
        logging.error('Unexpected thing has taken place, have a look in main', exc_info= True)
        raise e
    
if __name__ == '__main__':
    main()