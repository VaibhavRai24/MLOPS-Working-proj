import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
from src.logger import logging
import os


mlflow.set_tracking_uri("https://dagshub.com/VaibhavRai24/MLOPS-Working-proj.mlflow")
dagshub.init(repo_owner='VaibhavRai24', repo_name='MLOPS-Working-proj', mlflow=True)


def load_model(model_path:str):
    """
    Loads the trained model from the file
    
    """
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except FileNotFoundError:
        logging.error("Model file  not found")
        return None
    except Exception as e:
        logging.error(f"An error occured at the time of loading the model file")
        raise e
    
def load_data(file_path:str) -> pd.DataFrame:
    """
    Loads the data from the specified csv file
    
    """
    try:
        
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error("Data file not found")
        return None
    except Exception as e:
        logging.error(f"An error occured at the time of loading the data file")
        raise e
    
def evaluate_model(model, X_test:np.ndarray, y_test:np.ndarray) ->dict:
    """
    Evaluates the model on the test data and returns the evaluation metrics
    
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_probaa = model.predict_proba(X_test)[:,1]
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_probaa)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'auc': auc
        }
    
        logging.info(f"Model has been evaluated successfully")
        return metrics
    except Exception as e:
        logging.error(f"An error occured at the time of evaluating the model")
        raise e
    
    
def save_metrices(metrics:dict, file_path:str)-> None:
    """
    Saves the evaluation metrics to a json file
    
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file)
        logging.info(f"Metrics have been saved successfully")
    except Exception as e:
        logging.error(f"An error occured at the time of saving the metrics")
        raise e

    
def save_model_information(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
    
    
def main():
    mlflow.set_experiment("Running pipeline")
    with mlflow.start_run() as run:
        try:
            model = load_model("models/model.pkl")
            data = load_data('./data/processed/test_bow.csv')
            X_test = data.iloc[:, :-1].values
            y_test = data.iloc[:, -1].values
            
            metrices = evaluate_model(model, X_test, y_test)
            save_metrices(metrices, 'reports/metrics.json')
            
            for metric_name, metric_value in metrices.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(model, "model")
            save_model_information(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')
            
        except Exception as e:
            logging.error(f"An error occured at the time of running the model evaluation script")
            raise e
        
if __name__ == '__main__':
    main()