from lightgbm import LGBMRegressor
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib, pandas as pd, os

df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

print("LightGBM Results:", scores)
os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
joblib.dump(model, r'D:\trainDelayPrediction\models\lgbm_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\lgbm_metrics.csv', index=False)
