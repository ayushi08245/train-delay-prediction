import matplotlib.pyplot as plt
import pandas as pd

# === Step 1: Define model performance data ===
data = {
    'Model': [
        'CatBoost', 'ElasticNet', 'XGBoost', 'LightGBM',
        'SVR', 'MLP', 'Random Forest', 'KNN'
    ],
    'MAE': [27.87, 84.13, 29.15, 29.06, 76.15, 32.35, 31.27, 35.88],
    'RMSE': [41.64, 130.16, 47.73, 48.26, 202.18, 50.06, 54.78, 54.68],
    'R2': [0.960, 0.610, 0.948, 0.946, 0.059, 0.942, 0.931, 0.931]
}
df = pd.DataFrame(data)

# === Define a color palette ===
colors = plt.cm.tab10.colors  # 10 distinct colors

# === Step 2: Vertical bar plot for MAE ===
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['MAE'], color=colors)
plt.ylabel('Mean Absolute Error')
plt.title('Model Comparison - MAE')
plt.xticks(rotation=45)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('mae_barplot.png')
plt.show()

# === Step 3: Vertical bar plot for R² ===
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['R2'], color=colors)
plt.ylabel('R² Score')
plt.title('Model Comparison - R² Score')
plt.xticks(rotation=45)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('r2_barplot.png')
plt.show()

# === Step 4: Vertical bar plot for RMSE ===
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['RMSE'], color=colors)
plt.ylabel('Root Mean Squared Error')
plt.title('Model Comparison - RMSE')
plt.xticks(rotation=45)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('rmse_barplot.png')
plt.show()
