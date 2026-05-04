# src/train.py
import os
import logging
import polars as pl
import pandas as pd
import mlflow
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fallback για παλιές εκδόσεις sklearn
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

# ==========================================
# ⚙️ SETUP
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
DATA_DIR = os.environ["DATA_DIR"]

# ==========================================
# 🛠️ HELPERS
# ==========================================

def load_data():
    logging.info(f"📂 Loading data from {DATA_DIR}...")
    return (
        pl.read_parquet(f"{DATA_DIR}/feature_matrix_train.parquet"),
        pl.read_parquet(f"{DATA_DIR}/feature_matrix_val.parquet"),
        pl.read_parquet(f"{DATA_DIR}/feature_matrix_test.parquet")
    )

def prepare_data(df):
    df_pd = df.to_pandas()
    if "country_code" in df_pd.columns:
        df_pd["country_code"] = df_pd["country_code"].astype("category")
    
    drop_cols = ["period_start", "price_eur_mwh"]
    cols_to_drop = [col for col in drop_cols if col in df_pd.columns]
    X = df_pd.drop(columns=cols_to_drop)
    y = df_pd["price_eur_mwh"]
    return X, y

def evaluate(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = np.abs(y_true) > 1.0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))
    else:
        mape = 0.0
        
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

# ==========================================
# 🚀 TRAINING & LOGGING
# ==========================================

def train_and_log(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        logging.info(f"⚡ Training {model_name}...")

        feature_names = list(X_train.columns)
        mlflow.log_params(model.get_params())

        if "lightgbm" in model_name.lower():
            cat_features = [
                i for i, col in enumerate(feature_names) 
                if X_train[col].dtype.name == 'category'
            ]
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)],
                categorical_feature=cat_features if cat_features else None,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        test_pred = model.predict(X_test)
        metrics = evaluate(y_test.to_numpy(), test_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            import json
            mlflow.log_dict(importance_df.head(20).to_dict(orient="records"), "feature_importance.json")

        if "lightgbm" in model_name.lower():
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.xgboost.log_model(model, artifact_path="model")

        logging.info(f"✅ {model_name} Metrics: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}")

# ==========================================
# 🏁 MAIN
# ==========================================

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        train_raw, val_raw, test_raw = load_data()
        X_train, y_train = prepare_data(train_raw)
        X_val, y_val = prepare_data(val_raw)
        X_test, y_test = prepare_data(test_raw)

        models = {
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=1000, 
                learning_rate=0.05, 
                random_state=42, 
                verbose=-1
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=1000, 
                learning_rate=0.05, 
                early_stopping_rounds=50,
                eval_metric="rmse",
                tree_method="hist", 
                enable_categorical=True, 
                random_state=42
            )
        }

        for name, model in models.items():
            try:
                train_and_log(model, name, X_train, y_train, X_val, y_val, X_test, y_test)
            except Exception as e:
                logging.error(f"❌ Failed training {name}: {e}")

        logging.info("\n🚀 Pipeline finished! Check MLflow UI.")

    except Exception as e:
        logging.error(f"💥 Critical Error: {e}")

if __name__ == "__main__":
    main()