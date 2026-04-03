from sklearn.neighbors import KNeighborsRegressor
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib, pandas as pd, os

df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = KNeighborsRegressor(n_neighbors=5)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

print("KNN Results:", scores)
os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
joblib.dump(model, r'D:\trainDelayPrediction\models\knn_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\/preprocessing_pipeline.pkl')
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\knn_metrics.csv', index=False)
