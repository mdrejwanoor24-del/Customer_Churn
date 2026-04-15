# 📉 Customer Churn Prediction using Machine Learning

Churn Prediction project that identifies customers who are likely to discontinue a service.
The model is trained using classical machine learning algorithms and deployed as an interactive Streamlit web application.

🔗 Live App:
👉 https://churnprediction-afe4bt8gfrljqjvribprqy.streamlit.app/

## 📌 Problem Statement
Customer churn is a critical issue for subscription-based businesses.
Predicting churn in advance allows organizations to:
- Retain high-value customers
- Reduce revenue loss
- Design targeted retention strategies

This project predicts whether a customer is likely to Churn or Not Churn based on historical customer data.

## 🚀 Key Features
- End-to-end ML pipeline
- Data cleaning & exploratory analysis using Jupyter
- Feature engineering & preprocessing
- Multiple ML models trained and compared
- Best-performing model selected for deployment
- Streamlit-based UI for real-time predictions
- Production-ready project structure

## 🌐 Live Application
🔗 Try the app here:
👉 https://churnprediction-afe4bt8gfrljqjvribprqy.streamlit.app/

### App Capabilities
- Enter customer details
- Predict Churn / No Churn
- Fast, real-time inference

## 🧠 Machine Learning Workflow
```
Raw Customer Data
   ↓
Data Cleaning & Feature Engineering
   ↓
Exploratory Data Analysis (EDA)
   ↓
Model Training & Evaluation
   ↓
Best Model Selection
   ↓
Streamlit Deployment
```

## 📁 Project Structure
```
customer_churn/
│
├── app.py                          # Streamlit entry point (ROOT)
├── requirements.txt                # Dependencies (ROOT)
├── README.md
├── .gitignore
│
├── src/                            # Core ML logic
│   ├── __init__.py
│   ├── config.py                   # Paths, constants
│   ├── preprocessing.py            # Data preprocessing logic
│   ├── training.py                 # Model training pipeline
│   └── deployment.py               # Model loading & prediction service
│
├── data/                           # Data files
│   ├── Telco_Customer_Churn.csv    # Raw dataset
│   └── cleaned_data.csv            # Cleaned dataset (from notebook)
│
├── notebooks/                      # Analysis & experimentation
│   ├── data_cleaning.ipynb
│   └── eda.ipynb
│
├── models/                         # Trained models & artifacts
│   ├── best_model.joblib
│   ├── gradient_boosting_best_model.joblib
│   ├── log_reg_best_model.joblib
│   ├── random_forest_best_model.joblib
│   ├── svc_best_model.joblib
│   └── feature_columns.json
│
└── venv/                           # Virtual environment (ignored)


```
## 📊 Dataset
- Telco_Customer_Churn.csv – Raw customer churn dataset
- cleaned_data.csv – Cleaned dataset created using Jupyter Notebook

The dataset contains:
  - Demographic information
  - Account details
  - Service usage metrics

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
  git clone https://github.com/yourusername/customer_churn.git
  cd customer_churn
  
  ### 2️⃣ Create and activate a virtual environment
  - python -m venv venv
  - venv\Scripts\activate      # Windows
  - source venv/bin/activate   # macOS/Linux
  
### 3️⃣ Install dependencies
  - pip install -r requirements.txt

## 🧹 Data Preparation
- Data cleaning and EDA are performed in:
    - notebooks/data_cleaning.ipynb
    - notebooks/eda.ipynb
- After cleaning, export the processed dataset to:
    - data/cleaned_data.csv
  The training pipeline expects this file to exist

## 🤖 Model Training
Run training from the project root:
   - python -m src.training
Training will:
  - Load cleaned data
  - Create train/test splits (stratified)
  - Build preprocessing pipeline
  - Train multiple ML models using GridSearchCV
  - Save:
    - models/best_model.joblib
    - models/<model>_best_model.joblib
    - models/feature_columns.json
  
  The saved model and feature file ensure consistent prediction during deployment.

## 🖥️ Run Streamlit App
From the project root:
   - streamlit run src/app.py

## Features:
  - Single-customer prediction
  - Batch predictions via CSV upload
  - Automatic feature alignment
  - Probability output
  - Clean UI with error handling
  
## 📦 Deployment & Prediction API
### Deployment:
   - src/deployment.py

### Prediction:
  - predict_single(input_dict)   # returns dict with prediction + probability
  - predict_batch(dataframe)     # returns dataframe with predictions appended
  - The functions: (Align input to training features)
    - Coerce numeric types
    - Handle missing columns gracefully
    - Ensure stable predictions

## 📊 Technologies Used
- Python 3.10+
- Pandas, NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🚀 Future Enhancements
- SHAP explainability
- FastAPI REST API
- Docker deployment
- Monitoring & model drift detection
- Optuna Bayesian optimization

## 🤝 Contributing
- Contributions, suggestions, and feature requests are welcome.
- Feel free to open an issue or submit a pull request.

## 🙏 Acknowledgements
- Inspired by public telecom churn datasets.
- Thanks to the open-source community for their amazing tools.
