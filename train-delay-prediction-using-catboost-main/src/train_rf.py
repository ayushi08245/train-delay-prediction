from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib
import pandas as pd

df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

print("Random Forest Results:", scores)
joblib.dump(model, r'D:\trainDelayPrediction\models\rf_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')

pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\rf_metrics.csv', index=False)
