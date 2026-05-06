import os
import logging
import polars as pl
import pandas as pd
import mlflow
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# ==========================================
# SETUP
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
DATA_DIR = os.environ["DATA_DIR"]
MODEL_NAME = "energy_price_model"
PREDICTION_EXPERIMENT = "energy_price_predictions"

# ==========================================
# HELPERS
# ==========================================

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        model_uri = f"models:/{MODEL_NAME}@champion"
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model loaded from Registry: {MODEL_NAME}@champion")
        return model, "registry"
        
    except Exception:
        logging.warning("Registry not found, falling back to best run...")
        
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            order_by=["metrics.test_rmse ASC"],
            max_results=1
        )
        
        if runs.empty:
            raise RuntimeError(f"No runs found in experiment '{EXPERIMENT_NAME}'")
        
        run_id = runs.iloc[0]["run_id"]
        run_name = runs.iloc[0]["tags.mlflow.runName"]
        
        try:
            model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
            flavor = "lightgbm"
        except Exception:
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            flavor = "xgboost"
            
        logging.info(f"Model loaded from run: {run_id} ({run_name}) [{flavor}]")
        return model, "run"


def load_input_data(filepath: str) -> pl.DataFrame:
    logging.info(f"Loading input data from {filepath}...")
    return pl.read_parquet(filepath)


def prepare_input(df: pl.DataFrame) -> pd.DataFrame:
    df_pd = df.to_pandas()
    
    if "country_code" in df_pd.columns:
        df_pd["country_code"] = df_pd["country_code"].astype("category")
    
    drop_cols = ["period_start", "price_eur_mwh"]
    cols_to_drop = [col for col in drop_cols if col in df_pd.columns]
    X = df_pd.drop(columns=cols_to_drop)
    
    return X


def log_predictions(predictions: np.ndarray, input_data: pd.DataFrame, model_source: str) -> pd.DataFrame:
    results_df = pd.DataFrame({
        "prediction_eur_mwh": predictions,
        "timestamp": datetime.now().isoformat(),
        "model_source": model_source
    })
    
    mlflow.log_dict(
        results_df.head(100).to_dict(orient="records"),
        f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    logging.info("Predictions logged as artifact.")
    
    return results_df

# ==========================================
# PREDICTION
# ==========================================

def predict(input_filepath: str, output_filepath: str = None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(PREDICTION_EXPERIMENT)
    
    model, source = load_model()
    
    input_df = load_input_data(input_filepath)
    X = prepare_input(input_df)
    
    logging.info(f"Making predictions on {len(X)} samples...")
    predictions = model.predict(X)
    logging.info(f"Done. Range: [{predictions.min():.2f}, {predictions.max():.2f}] EUR/MWh")
    
    with mlflow.start_run(run_name=f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_source", source)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("input_samples", len(X))
        mlflow.log_metric("prediction_mean", float(np.mean(predictions)))
        mlflow.log_metric("prediction_std", float(np.std(predictions)))
        mlflow.log_metric("prediction_min", float(np.min(predictions)))
        mlflow.log_metric("prediction_max", float(np.max(predictions)))
        
        results_df = log_predictions(predictions, X, source)
    
    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        results_df.to_csv(output_filepath, index=False)
        logging.info(f"Results saved to {output_filepath}")
    
    return predictions, results_df

# ==========================================
# MAIN
# ==========================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Energy Price Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV (optional)")
    args = parser.parse_args()
    
    try:
        predictions, results = predict(args.input, args.output)
        logging.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()