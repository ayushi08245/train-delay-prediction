from sklearn.neural_network import MLPRegressor
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib, pandas as pd, os

df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

print("MLP Results:", scores)
os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
joblib.dump(model, r'D:\trainDelayPrediction\models\mlp_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\mlp_metrics.csv', index=False)
