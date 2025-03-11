# Sentiment Analysis using MLOps

## Overview
This project implements a **Sentiment Analysis** system while integrating MLOps best practices. The entire pipeline is automated, ensuring scalability, reproducibility, and efficiency. The solution includes **data ingestion, preprocessing, model training, deployment**, and monitoring using various MLOps tools.

## Tech Stack
- **Machine Learning**: Scikit-learn, TensorFlow/PyTorch
- **Data Handling**: Pandas, NumPy, NLTK, Spacy
- **Data Versioning**: DVC (Data Version Control)
- **Containerization**: Docker
- **Cloud Storage & Deployment**: AWS (S3, ECR, Lambda, Sagemaker)
- **Orchestration & Scaling**: Kubernetes

## Project Workflow

### 1. **Data Ingestion from AWS S3**
- Raw data (e.g., customer reviews, tweets) is stored in **AWS S3**.
- A pipeline is set up to **fetch data** from S3 automatically.
- Data is version-controlled using **DVC**.

### 2. **Data Preprocessing**
- Text cleaning (removing stopwords, punctuation, stemming, and lemmatization).
- Tokenization and vectorization using **TF-IDF, Word2Vec, or BERT embeddings**.
- Splitting into training and validation datasets.

### 3. **Model Training & Evaluation**
- Train sentiment analysis models using **Logistic Regression, LSTMs, or Transformer-based models**.
- Hyperparameter tuning with **GridSearchCV/Optuna**.
- Model evaluation using **accuracy, precision, recall, F1-score**.

### 4. **Data Versioning with DVC**
- Track changes in datasets and models with **DVC**.
- Store and retrieve different versions of data from **AWS S3**.

### 5. **Containerization using Docker**
- Create a **Docker container** for the model to ensure environment consistency.
- The container includes necessary dependencies (ML model, preprocessing pipeline, API setup).

### 6. **Model Storage and Deployment using AWS**
- Trained models are stored in **AWS S3**.
- Model deployment using **AWS Lambda, SageMaker, or EC2 instances**.

### 7. **Orchestration with Kubernetes**
- Deploy model services using **Kubernetes clusters** for auto-scaling.
- Kubernetes ensures **fault tolerance** and **load balancing**.

### 8. **Continuous Integration & Monitoring**
- CI/CD pipeline set up using **GitHub Actions or Jenkins**.
- Monitor model performance with **Prometheus & Grafana**.

## Setup Instructions
### Prerequisites
- Install Docker, DVC, AWS CLI, and Kubernetes.
- Configure AWS credentials (`aws configure`).
- Clone this repository:
  ```sh
  git clone https://github.com/your-repo/sentiment-analysis-mlops.git
  cd sentiment-analysis-mlops
  ```

### Running the Pipeline
1. **Set up virtual environment**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Run data ingestion**:
   ```sh
   python src/data_ingestion.py
   ```
3. **Track data using DVC**:
   ```sh
   dvc add data/raw
   dvc push
   ```
4. **Train the model**:
   ```sh
   python src/train.py
   ```
5. **Build Docker container**:
   ```sh
   docker build -t sentiment-analysis .
   ```
6. **Deploy with Kubernetes**:
   ```sh
   kubectl apply -f deployment.yaml
   ```

## Folder Structure
```
📂 sentiment-analysis-mlops
├── 📁 data                # Raw and processed data
├── 📁 models              # Saved models
├── 📁 src                 # Source code
│   ├── data_ingestion.py  # Fetches data from AWS S3
│   ├── preprocess.py      # Cleans and transforms text
│   ├── train.py           # Trains the ML model
│   ├── predict.py         # API for model inference
├── 📁 kubernetes          # Kubernetes deployment files
├── 📁 docker              # Docker-related files
├── requirements.txt       # Python dependencies
├── dvc.yaml               # DVC pipeline config
├── deployment.yaml        # Kubernetes deployment script
└── README.md              # Project documentation
```

## Future Improvements
- Implement **active learning** to continuously improve the model.
- Integrate **MLOps tools like MLflow** for experiment tracking.
- Add **real-time sentiment analysis using Kafka and Spark**.

## Contributors
- **Your Name** - Vaibhav rai

## License
This project is licensed under the MIT License.