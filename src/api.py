import os
import logging
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
import mlflow

# ==========================================
# SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Εδώ λέμε στο API να κοιτάζει τοπικά
MLFLOW_TRACKING_URI = "./mlruns"
# Εδώ κρατάμε το localhost:5433 γιατί η βάση είναι στο Docker
DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/energy_market" 

MODEL_NAME = "energy_price_model"

# Η ΑΚΡΙΒΗΣ ΔΙΑΔΡΟΜΗ ΠΟΥ ΕΙΔΑΜΕ ΣΤΟ TREE ΣΟΥ
LOCAL_MODEL_PATH = r"C:\Users\USER\Desktop\A_Programming_Examples\Coding\energy-mlops\mlruns\294445254527345384\models\m-cda928d0b5764e31970f047f48fec241\artifacts"
model = None
model_meta = {}

# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_meta
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        logger.info(f"Loading model directly from: {LOCAL_MODEL_PATH}")
        # ΦΟΡΤΩΝΟΥΜΕ ΑΠΕΥΘΕΙΑΣ ΑΠΟ ΤΟΝ ΦΑΚΕΛΟ
        model = mlflow.pyfunc.load_model(LOCAL_MODEL_PATH)
        
        model_meta = {
            "name": MODEL_NAME,
            "alias": "champion",
            "status": "ready",
            "source": "local_disk"
        }
        logger.info("Model loaded successfully from local disk!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        # Αν αποτύχει, δεν κλείνουμε το API για να μπορείς να δεις το /health
        model = None 
    yield

# ==========================================
# APP
# ==========================================
app = FastAPI(
    title="Energy Price Prediction API",
    description="API για πρόβλεψη τιμών ενέργειας",
    version="1.0.0",
    lifespan=lifespan
)

# [Τα υπόλοιπα SCHEMAS, HELPERS και ENDPOINTS μένουν ίδια...]
# (Κράτα τον κώδικα για PredictionRequest, PredictionResponse, prepare_input, save_to_db κλπ όπως τον είχες)

class PredictionRequest(BaseModel):
    country_code: str = Field(..., description="Country code (e.g. 'GR')")
    load_mw: float = Field(..., description="Load forecast in MW")
    Solar: float = Field(..., description="Solar generation in MW")
    Wind: float = Field(..., description="Wind generation in MW")
    Coal: float = Field(..., description="Coal generation in MW")
    Gas: float = Field(..., description="Gas generation in MW")
    Hydro: float = Field(..., description="Hydro generation in MW")
    Biomass: float = Field(..., description="Biomass generation in MW")
    Nuclear: float = Field(..., description="Nuclear generation in MW")
    Geothermal: float = Field(..., description="Geothermal generation in MW")
    Other: float = Field(..., description="Other generation in MW")
    hour: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    dayofweek: int = Field(..., description="Day of week (0=Monday)", ge=0, le=6)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    is_weekend: int = Field(..., description="Is weekend (0 or 1)", ge=0, le=1)
    lag_24h: float = Field(..., description="Price 24h ago")
    lag_168h: float = Field(..., description="Price 168h ago (1 week)")
    price_mean_24h: float = Field(..., description="Mean price last 24h")
    load_mean_7d: float = Field(..., description="Mean load last 7 days")
    es_price_cap_flag: int = Field(..., description="Price cap flag (0 or 1)", ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction_eur_mwh: float
    model_name: str
    model_alias: str
    timestamp: str
    status: str

def prepare_input(request: PredictionRequest) -> pd.DataFrame:
    df = pd.DataFrame([request.model_dump()])
    df["country_code"] = df["country_code"].astype("category")
    return df

def save_to_db(request: PredictionRequest, prediction: float):
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.DataFrame([{
            **request.model_dump(),
            "prediction_eur_mwh": prediction,
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "model_alias": "champion"
        }])
        df.to_sql("api_predictions", con=engine, if_exists="append", index=False)
        logger.info("Prediction saved to DB.")
    except Exception as e:
        logger.warning(f"Could not save to DB: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = prepare_input(request)
        prediction = float(model.predict(df)[0])
        save_to_db(request, prediction)
        return PredictionResponse(
            prediction_eur_mwh=round(prediction, 2),
            model_name=MODEL_NAME,
            model_alias="champion",
            timestamp=datetime.now().isoformat(),
            status="success"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))