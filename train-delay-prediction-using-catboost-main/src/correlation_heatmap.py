import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r'data/train delay data.csv')  # Replace with your actual file name

# Optional: Preview the data
print("Sample data:\n", df.head())

# Step 1: Encode categorical columns
# Automatically detect and encode all object-type columns
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = df.copy()

for col in categorical_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Step 2: Compute correlation matrix
corr_matrix = df_encoded.corr()

# Step 3: Plot the heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

# Step 4: Add title and layout
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()
