import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle


def load_params(params_path:str) ->dict:
    """
    Basically this loads the params from the yaml file
    
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters are retrevied from the given file path', params_path)
        return params
    
    except FileNotFoundError:
        logging.error('File not found at the given point', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('File not found or the file is empty', params_path)
        raise e
    except Exception as e:
        logging.error('Unexpected thing has takeen place, have a look')
        raise e
    
    
def load_data(file_path: str) -> pd.DataFrame:
    """
    Basically this function used to load the data from the file
    
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna(" ", inplace= True)
        logging.info('data has been loaded and the Nan has been filed',file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the csv file ')
        raise
    except Exception as e:
        logging.error('Unexcepted thing has been taken place , have a look')
        raise
    
def apply_the_bow(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int) ->tuple:
    """
    Baically this function is used to apply the bag of words technique to convert the categorical to numerical form
    """
    try:
        logging.info('Applying the bow........................>>>>>>>>>>>>>>>>>>')
        vectorizer = CountVectorizer(max_features= max_features)
        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.fit_transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        logging.info('Bag of words applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        logging.error('Error during Bag of Words transformation', e)
        raise 
    
def save_data(df:pd.DataFrame, file_path:str) -> None:
    """
    save the data into the CSV file
    
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        df.to_csv(file_path, index= False)
        logging.info('Data saved to ', file_path)
    except Exception as e:
        logging.error('Unexcepted error occurred during the process while saving the data')
        raise
    
def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        train_df, test_df = apply_the_bow(train_data= train_data, test_data= test_data, max_features= max_features)
        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
        
    except Exception as e:
        logging.error('Failed to complete the feature engineering process')
        
if __name__ == '__main__':
    main()