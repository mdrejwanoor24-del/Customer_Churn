 #load pre-trained data
#imputing & Encoding
# Train test split
# Model training & Hyperparameter tuning
#model evaluation & select best model
#dump best model

# Imports
import joblib
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.impute import SimpleImputer   
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
#Load Pre_Cleaned Data
CLEANED_DATA_PATH = r"C:\Users\DELL\Downloads\Customer_Churn_Prediction\Data\cleaned_data.csv"
data = pd.read_csv(CLEANED_DATA_PATH)
print(data.head())
print("Data Load Successful")
# Data Back-up
df = data.copy()

# Target & Features
x= df.drop(columns=["Churn","Churn_Encoded"], axis=1)
y = df["Churn_Encoded"]
num_cols = x.select_dtypes(include="number").columns.to_list()
cat_cols = x.select_dtypes(include="object").columns.to_list()


# Imputing & Encoding
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))])
preprocessor = ColumnTransformer(transformers=[
    ('num_transformer', num_pipeline, num_cols),
    ('cat_transformer', cat_pipeline, cat_cols)])
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# MOdel Dictionary  
models = {
    'Logistic Regression':(
        LogisticRegression(max_iter=1000),
         {'model__C': [0.01, 0.1, 1]}
    ),
    'Random Forest':(
        RandomForestClassifier(),
        {'model__n_estimators': [50,100, 200],
        'model__max_depth': [None,5,10,20]
         
         }),
         "Adaboost":(
             AdaBoostClassifier(),
             {'model__n_estimators': [50, 100, 200]}
         ),
         "XGBoost":(
                XGBClassifier(evaluation_metric='logloss'),
                {'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2]
                }),
            "SVM":(
                SVC(),
                {'model__C': [0.01, 0.1, 1],
                'model__kernel': ['linear', 'rbf']}
            )
}
result = []
best_model = None
best_score = 0  
# Model Training 
for model_name, (model, params) in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    grid = GridSearchCV(pipeline,
                        param_grid= params, 
                        cv=5, scoring='f1',
                          n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    result.append({
        'Model': model,
        'Accuracy': accuracy,
        'F1 Score': f1
    })
    if f1 > best_score:
        best_score = f1
        best_model = grid.best_estimator_
        result_df = pd.DataFrame(result)
        #Short dataframe by F1 Score
        result_df = result_df.sort_values(by='F1 Score', ascending=False)
        print('model comparison :\n')
        
print(result_df)
print('nBest Model :', result_df.iloc[0]['Model'])
MODEL_PATH = r"C:\Users\DELL\Downloads\Customer_Churn_Prediction\best_model.pkl"
#dump the model
joblib.dump(best_model, MODEL_PATH)
print("Model Trained and Saved Successfully")



        
