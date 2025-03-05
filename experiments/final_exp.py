import os
import re
import string
import dagshub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "https://dagshub.com/VaibhavRai24/MLOPS-Working-proj.mlflow"
dagshub.init(repo_name= "MLOPS-Working-proj", repo_owner= "VaibhavRai24", mlflow= True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("final_exp_finetuning")

#________________________________________________________________>

def preprocessing_text(text):
    """
    Basically in this step we are applying multiple preprocessing steps on the text data.
    
    """
    lemmitizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = " ".join([lemmitizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    return text.strip()

#________________________________________________________________>


def load_and_prepare_data(filepath):
    """
    This function is used to load the data and then apply the preprocessing steps on the text data.
    
    """
    df = pd.read_csv(filepath)
    df["review"] = df["review"].astype(str).apply(preprocessing_text)
    
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    
    vectorizers = TfidfVectorizer(max_features=500)
    X = vectorizers.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizers


#________________________________________________________________>


def train_model(X_train, X_test, y_train, y_test, vectorizers):
    """
    This function is used to train the model on the given data.
    
    """
    params_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_searching = GridSearchCV(LogisticRegression(), param_grid=params_grid, cv=5, scoring= "f1", n_jobs= -1)
        grid_searching.fit(X_train, y_train)
        y_pred = grid_searching.predict(X_test)
        
        for params, mean_score, std_score in zip(grid_searching.cv_results_["params"], 
                                                 grid_searching.cv_results_["mean_test_score"], 
                                                 grid_searching.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        
            metrcies = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "mean_cv_score": mean_score,
                "std_cv_score": std_score
            }
            
            mlflow.log_params(params)
            mlflow.log_metrics(metrcies)
            
            print(f"Params: {params} | Accuracy: {metrcies['accuracy']:.4f} | F1: {metrcies['f1_score']:.4f}")
            
        best_params = grid_searching.best_params_
        best_model = grid_searching.best_estimator_
        best_f1  = grid_searching.best_score_
        
        
        mlflow.log_params(best_params)
        mlflow.log_metrics({"best_f1": best_f1})
        mlflow.sklearn.log_model(best_model, "model")   
        print(f"Best Params: {best_params} | Best F1: {best_f1:.4f}")
        
        
        
#________________________________________________________________>

if __name__ == "__main__":
    data_path = r"C:\Users\VAIBHAVRAI\OneDrive\Desktop\mlops-main\MLOPS-Working-proj\experiments\real_data.csv"
    (X_train, X_test, y_train, y_test), vectorizers = load_and_prepare_data(data_path)
    train_model(X_train, X_test, y_train, y_test, vectorizers)