import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_rel

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

from preprocess import load_data, preprocess_data
# Note: split_data not needed here since we use CV

# Load and preprocess data
df = load_data()
X, y, pipeline = preprocess_data(df)

# Define models
models = {
    "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_seed=42),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "RandomForest": RandomForestRegressor(),
    "MLP": MLPRegressor(max_iter=500),
    "KNN": KNeighborsRegressor(),
    "ElasticNet": ElasticNet(),
    "SVR": SVR()
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {"MAE": [], "RMSE": [], "R2": []} for name in models}

# Run CV
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        results[name]["RMSE"].append(mean_squared_error(y_test, y_pred, squared=False))
        results[name]["R2"].append(r2_score(y_test, y_pred))

# Summarize results
summary = []
catboost_mae = results["CatBoost"]["MAE"]

for name, metrics in results.items():
    mae_mean, mae_std = np.mean(metrics["MAE"]), np.std(metrics["MAE"])
    rmse_mean, rmse_std = np.mean(metrics["RMSE"]), np.std(metrics["RMSE"])
    r2_mean, r2_std = np.mean(metrics["R2"]), np.std(metrics["R2"])
    
    # Paired t-test vs CatBoost
    if name != "CatBoost":
        t_stat, p_val = ttest_rel(catboost_mae, metrics["MAE"])
        significance = "✔ Significant" if p_val < 0.05 else "Not Significant"
        p_val_str = f"{p_val:.4f}"
    else:
        p_val_str = "—"
        significance = "—"
    
    summary.append([name, f"{mae_mean:.2f} ± {mae_std:.2f}", 
                    f"{rmse_mean:.2f} ± {rmse_std:.2f}", 
                    f"{r2_mean:.3f} ± {r2_std:.3f}", 
                    p_val_str, significance])

# Create DataFrame
df_results = pd.DataFrame(summary, columns=["Model", "MAE", "RMSE", "R²", "p-value vs CatBoost", "Significance"])
print("\n=== Statistical Testing Results ===")
print(df_results)

# Save results
df_results.to_csv(r'D:\trainDelayPrediction\reports\model_comparison_with_pvalues.csv', index=False)
