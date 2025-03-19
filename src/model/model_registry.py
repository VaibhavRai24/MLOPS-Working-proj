import json
import mlflow
import logging
import os
import dagshub
from src.logger import logging
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


mlflow.set_tracking_uri("https://dagshub.com/VaibhavRai24/MLOPS-Working-proj.mlflow")
dagshub.init(repo_owner='VaibhavRai24', repo_name='MLOPS-Working-proj', mlflow=True)

def load_model_infos(model_path:str) ->dict:
    """
    Loads the model info from the JSON file
    
    """
    try:
        with open(model_path, 'r') as file:
            model_info = json.load(file)
        logging.info("Model info loaded successfully")
        return model_info
    except FileNotFoundError:
        logging.error("Model info file not found")
        return None
    except Exception as e:
        logging.error(f"An error occured at the time of loading the model info file")
        raise e
    
def register_model(model_name:str, model_infos:dict):
    """
    Registers the model in the model registry
    
    """
    try:
        model_uri = f"runs:/{model_infos['run_id']}/{model_infos['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.info("Model registered successfully")
    except Exception as e:
        logging.error("An error occured at the time of registering the model")
        raise e
    
def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_infos = load_model_infos(model_info_path)
        model_name  = "running_model"
        register_model(model_name, model_infos)
    except Exception as e:
        logging.error("An error occured at the time of registering the model")
        raise e
    
if __name__ == '__main__':
    main()