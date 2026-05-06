import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

runs = mlflow.search_runs(experiment_names=["energy_price_forecasting"])
if runs.empty:
    print("❌ No runs found")
else:
    run_id = runs.iloc[0]["run_id"]
    model_name = runs.iloc[0]["tags.mlflow.runName"]
    print(f"🔍 Trying to load {model_name} from run {run_id}")
    
    try:
        if "lightgbm" in model_name.lower():
            model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        else:
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        print("✅ Model loaded via API → artifacts exist!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")