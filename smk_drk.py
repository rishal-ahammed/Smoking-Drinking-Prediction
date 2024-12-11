# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("D:\ML_lnkng\cleaned.csv")

# Initialize a dictionary to store models
models = {}

# Loop through the target columns (-2 and -1)
for target_col in [-2, -1]:
    print(f"\nEvaluating model for target variable: Column {target_col}")
    
    # Feature matrix and target variable
    X = df.iloc[:, :-2].values  # Features
    y = df.iloc[:, target_col].values  # Target variable

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Logistic Regression Model
    logreg = LogisticRegression(solver='liblinear', max_iter=1000)
    logreg.fit(X_train, y_train)

    # Save the model with a unique name
    models[f"target_{target_col}"] = logreg

    # Save the scaler for consistent preprocessing during inference
    with open(f"scaler_{target_col}.pkl", 'wb') as f:
        pickle.dump(scaler, f)

# Save all models into a single pickle file
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)
