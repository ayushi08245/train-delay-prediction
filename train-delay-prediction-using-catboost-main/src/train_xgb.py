from xgboost import XGBRegressor
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib
import pandas as pd
import os

# Load and preprocess data
df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

# Print and save results
print("XGBoost Results:", scores)

# Ensure folders exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# Save model and pipeline
joblib.dump(model, r'D:\trainDelayPrediction\models\xgb_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')

# Save metrics
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\xgb_metrics.csv', index=False)
