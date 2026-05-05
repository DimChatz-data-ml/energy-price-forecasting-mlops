import mlflow

# Σύνδεση στο MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Αναζήτηση runs
print("🔍 Searching for runs...")
runs = mlflow.search_runs(experiment_names=["energy_price_forecasting"])

if runs.empty:
    print("❌ No runs found. Check experiment name or MLflow connection.")
else:
    print("\n📊 Found runs:")
    print(runs[["run_id", "tags.mlflow.runName", "metrics.test_rmse"]].to_string())
    
    # Δοκιμή φόρτωσης μοντέλου (από το πρώτο run)
    run_id = runs.iloc[0]["run_id"]
    run_name = runs.iloc[0]["tags.mlflow.runName"]
    print(f"\n🔄 Loading model from run: {run_name} ({run_id})")
    
    try:
        if "lightgbm" in run_name.lower():
            model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        else:
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")