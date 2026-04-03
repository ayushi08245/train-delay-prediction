from sklearn.linear_model import ElasticNet
from preprocess import load_data, preprocess_data, split_data
from utils import evaluate_model
import joblib, pandas as pd, os

# Load and preprocess data
df = load_data()
X, y, pipeline = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train ElasticNet model
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
scores = evaluate_model(model, X_train, X_test, y_train, y_test)

# Print and save results
print("ElasticNet Results:", scores)
os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
joblib.dump(model, r'D:\trainDelayPrediction\models\elasticnet_model.pkl')
joblib.dump(pipeline, r'D:\trainDelayPrediction\models\preprocessing_pipeline.pkl')
pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\elasticnet_metrics.csv', index=False)
