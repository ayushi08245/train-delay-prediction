import pandas as pd
import joblib
import os

# === Paths ===
MODEL_PATH = r'D:\trainDelayPrediction\models\catboost_model_optimized.pkl'
PIPELINE_PATH = r'D:\trainDelayPrediction\models\catboost_preprocessing_pipeline.pkl'
INPUT_DATA_PATH = r'D:\trainDelayPrediction\data\new_schedule.csv'  # Replace with your actual input file
OUTPUT_PATH = r'D:\trainDelayPrediction\reports\predicted_delays.csv'

# === Step 1: Load model and preprocessing pipeline ===
model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# === Step 2: Load new raw input data ===
new_data = pd.read_csv(INPUT_DATA_PATH)

# === Step 3: Drop target column if present ===
if 'Historical Delay (min)' in new_data.columns:
    new_data = new_data.drop(columns=['Historical Delay (min)'])

# === Step 4: Apply preprocessing ===
X_new = pipeline.transform(new_data)

# === Step 5: Predict delays ===
predicted_delays = model.predict(X_new)

# === Step 6: Save predictions ===
new_data['Predicted_Delay'] = predicted_delays
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
new_data.to_csv(OUTPUT_PATH, index=False)

# === Step 7: Print sample predictions ===
print("✅ Prediction complete. Sample results:")
print(new_data[['Predicted_Delay']].head())
