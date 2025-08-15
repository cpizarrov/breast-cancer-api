from typing import Dict, List
import math
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
import uvicorn

MODEL_PATH = "model/breast_cancer_logreg.joblib" 

try:
    ARTIFACT = joblib.load(MODEL_PATH)
    MODEL = ARTIFACT["model"]           
    FEATURE_NAMES: List[str] = ARTIFACT["feature_names"]
    THRESHOLD: float = float(ARTIFACT.get("threshold", 0.5))
    TARGET_MAPPING = ARTIFACT.get("target_mapping", {"B": 0, "M": 1})
    INV_MAP = {v: k for k, v in TARGET_MAPPING.items()}
    LOAD_ERROR = None
except Exception as e:
    MODEL, FEATURE_NAMES, THRESHOLD = None, [], 0.5
    TARGET_MAPPING, INV_MAP = {"B": 0, "M": 1}, {0: "B", 1: "M"}
    LOAD_ERROR = str(e)

app = FastAPI(
    title="Breast Cancer Logistic Regression API",
    version="3.1.0",
    description="API para predecir cáncer."
)

# ==============================
# Esquemas
# ==============================

class PredictBody(BaseModel):
    features: Dict[str, float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, float]):
        if not isinstance(v, dict) or not v:
            raise ValueError("El campo 'features' debe ser un objeto con pares {feature: valor}.")
        for k, val in v.items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"'{k}' debe ser numérico y finito.")
        return v


class PredictResponse(BaseModel):
    predicted_class: int
    prob_maligno: float
    label: str
    threshold_used: float
    missing_features: List[str] = []
    extra_features: List[str] = []

# ==============================
# Helpers
# ==============================

def ensure_loaded():
    if MODEL is None:
        raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {LOAD_ERROR}")

def order_and_validate(features: Dict[str, float]) -> pd.DataFrame:
    if not FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="feature_names no disponible en el artefacto.")

    expected = set(FEATURE_NAMES)
    got = set(features.keys())
    missing = sorted(list(expected - got))
    extra = sorted(list(got - expected))
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Las claves de 'features' no coinciden con las esperadas (usa /features).",
                "missing_features": missing,
                "extra_features": extra,
                "expected": FEATURE_NAMES
            }
        )

    row = [float(features[name]) for name in FEATURE_NAMES]
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    return df

# ==============================
# Endpoints
# ==============================

@app.get("/")
def home():
    return {"msg": "API OK. Visita /docs para la UI de Swagger."}

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "error",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "features_count": len(FEATURE_NAMES),
        "threshold_saved": THRESHOLD,
        "load_error": LOAD_ERROR,
    }

@app.get("/features")
def features():
    ensure_loaded()
    return {"feature_names": FEATURE_NAMES}

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictBody, confidence: float = Query(default=THRESHOLD, ge=0.0, le=1.0)):
    ensure_loaded()
    df = order_and_validate(body.features)
    proba = float(MODEL.predict_proba(df)[:, 1][0])
    pred = int(proba >= confidence)
    return PredictResponse(
        predicted_class=pred,
        prob_maligno=proba,
        label={0: "B", 1: "M"}.get(pred, str(pred)),
        threshold_used=float(confidence),
        missing_features=[],
        extra_features=[],
    )

# Arranque local
if __name__ == "__main__":
    # uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)