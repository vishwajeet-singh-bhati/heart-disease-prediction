import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Cleveland dataset (most commonly used)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load data
df = pd.read_csv(url, names=columns, na_values='?')

# Save to data folder
df.to_csv('data/heart_disease.csv', index=False)
print(f"Dataset downloaded! Shape: {df.shape}")
print(df.head())  # Fixed the closing parenthesis here