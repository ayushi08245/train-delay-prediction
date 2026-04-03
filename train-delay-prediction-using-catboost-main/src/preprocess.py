import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path=r'D:\trainDelayPrediction\data\train delay data.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=['Historical Delay (min)'])
    y = df['Historical Delay (min)']

    # Identify column types
    numeric_features = ['Distance Between Stations (km)']
    categorical_features = ['Weather Conditions', 'Day of the Week', 'Time of Day', 'Train Type', 'Route Congestion']

    # Preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Final pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    return X_processed, y, pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
