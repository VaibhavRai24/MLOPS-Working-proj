import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from nltk.tokenize import word_tokenize
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


CONFIGURATION ={
    "data_filepath": r"C:\Users\VAIBHAVRAI\OneDrive\Desktop\mlops-main\MLOPS-Working-proj\experiments\real_data.csv",
    "test_size": 0.2,
    "experiment_name": "bow_and_tfidf_experiment_v2",
    "dagshub_repository_name": "MLOPS-Working-proj", 
    "dagshub_owner_name": "VaibhavRai24",
    "mlflow_tracking_uri": "https://dagshub.com/VaibhavRai24/MLOPS-Working-proj.mlflow",
}


mlflow.set_tracking_uri(CONFIGURATION["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIGURATION["dagshub_owner_name"],repo_name=CONFIGURATION["dagshub_repository_name"], mlflow= True)
mlflow.set_experiment(CONFIGURATION["experiment_name"])



def lemmatization(text):
    words  = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def removing_stopwords(text):
    words = word_tokenize(text)
    stopwords_list = stopwords.words("english")
    return " ".join([word for word in words if word not in stopwords_list])

def removing_numerical_values(text):
    words= word_tokenize(text)
    return " ".join([word for word in words if not any (word.isdigit() for word in words)])

def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)



def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] =df['review'].apply(remove_punctuation)
        df['review'] = df['review'].apply(removing_stopwords)
        df['review'] = df['review'].apply(removing_numerical_values)
        df['review'] = df['review'].apply(lemmatization)
        df['review'] = df['review'].apply(removing_urls)
        return df
    except Exception as e:
        print(f"Error in normalizing text: {e}")
        return df
    
    
def load_data(data_filepath):
    try:
        df = pd.read_csv(data_filepath)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0}).infer_objects(copy = False)
        return df
    except Exception as e:
        print(f"Error in loading data: {e}")
        raise
    
    
ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

VECTORIZERS = {
    "count_vectorizer": CountVectorizer(),
    "tfidf_vectorizer": TfidfVectorizer()
}


def training_and_evaluating(df):
    with mlflow.start_run(run_name= 'All Algorithms runned') as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name= f"{algo_name} with {vec_name}", nested= True) as child_run:
                    try:
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIGURATION["test_size"], random_state=42)
                        
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIGURATION["test_size"]
                        })
                        
                        
                        model = algorithm
                        model.fit(X_train, y_train)
                        log_models_params(algo_name, model)
                        
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1": f1_score(y_test, y_pred)
                        }
                        mlflow.log_metrics(metrics)
                        
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))
                        
                        
def log_models_params(algo_name, model):
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log['C'] = model.C
        
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
        
        
    mlflow.log_params(params_to_log)
    
if __name__ == "__main__":
    df = load_data(CONFIGURATION["data_filepath"])
    training_and_evaluating(df)