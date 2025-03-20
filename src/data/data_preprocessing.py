import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
import string
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_dataframe(df, col = 'text'):
    """
    Preprocces the dataframe by removing stopwords, punctuation, and lemmatizing the text.
    
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    def preprocess_text(text):
        """ 
        Helper function to preprocess the text.
        
        """
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        text = ''.join([char for char in text if not char.isdigit()])
        
        text = text.lower()
        
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        
        text = text.replace('؛', "")
        
        text = re.sub('\s+', ' ', text).strip()
       
        text = " ".join([word for word in text.split() if word not in stop_words])
        
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        
        return text
    
    
    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])
    return df


def main():
    try:
        train_data = pd.read_csv("./data/raw/train_data.csv")
        test_data = pd.read_csv("./data/raw/test_data.csv")
        logging.info("Data loaded successfully for preprocessing -------------------------------->.")
        
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')
        
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_data.csv"), index=False)
        logging.info("Data saved successfully after preprocessing--------------------------->.")
        
    except Exception as e:
        logging.error("Unexpected error has taken place while data preprocssing:")
        print(f'Error: {e}')
        
if __name__ == "__main__":  
    main()        