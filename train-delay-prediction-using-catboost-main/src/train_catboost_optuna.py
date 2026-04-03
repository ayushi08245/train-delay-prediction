import optuna
from catboost import CatBoostRegressor
from preprocess import load_data, preprocess_data, split_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd, numpy as np, joblib, os

def objective(trial):
    # Load and preprocess
    df = load_data()
    X, y, pipeline = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'verbose': 0
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best RMSE:", study.best_value)
    print("Best Params:", study.best_params)

    # Retrain with best params
    df = load_data()
    X, y, pipeline = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model = CatBoostRegressor(**study.best_params)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    scores = {
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }

    print("Optimized CatBoost Results:", scores)

    # Save model and metrics
    os.makedirs(r'D:\trainDelayPrediction\models', exist_ok=True)
    os.makedirs(r'D:\trainDelayPrediction\reports', exist_ok=True)
    joblib.dump(best_model, r'D:\trainDelayPrediction\models\catboost_model_optimized.pkl')
    joblib.dump(pipeline, r'D:\trainDelayPrediction\models\catboost_preprocessing_pipeline.pkl')
    pd.DataFrame([scores]).to_csv(r'D:\trainDelayPrediction\reports\catboost_metrics_optimized.csv', index=False)
