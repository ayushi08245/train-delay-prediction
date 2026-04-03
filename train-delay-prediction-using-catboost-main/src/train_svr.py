from sklearn.svm import SVR
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib, pandas as pd, os

df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

print("SVR Results:", scores)
os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
joblib.dump(model, r'D:\trainDelayPrediction\models\svr_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\svr_metrics.csv', index=False)
