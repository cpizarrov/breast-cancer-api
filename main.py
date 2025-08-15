import joblib
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator
import uvicorn

# ==============================
# Config / Carga del modelo
# ==============================
MODEL_PATH = os.getenv("MODEL_PATH", "model/breast_cancer_logreg.joblib")

try:
    ARTIFACT: Dict = joblib.load(MODEL_PATH)
    MODEL = ARTIFACT["model"]                           # Pipeline (StandardScaler + LogisticRegression)
    FEATURE_NAMES: List[str] = ARTIFACT["feature_names"]  # ORDEN exacto esperado por el modelo
    THRESHOLD: float = float(ARTIFACT.get("threshold", 0.5))
    TARGET_MAPPING = ARTIFACT.get("target_mapping", {"B": 0, "M": 1})
    INV_MAP = {v: k for k, v in TARGET_MAPPING.items()}   # {0:"B", 1:"M"}
    LOAD_ERROR = None
except Exception as e:
    MODEL, FEATURE_NAMES, THRESHOLD = None, [], 0.5
    TARGET_MAPPING, INV_MAP = {"B": 0, "M": 1}, {0: "B", 1: "M"}
    LOAD_ERROR = str(e)

app = FastAPI(
    title="Breast Cancer Logistic Regression API",
    version="1.0.0",
    description="API para predecir Benigno (B) vs Maligno (M) usando un modelo serializado (joblib)."
)

# ==============================
# Esquemas Pydantic
# ==============================
class PredictBody(BaseModel):
    """Un solo caso por request. Debe contener TODAS las features esperadas."""
    features: Dict[str, float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, float]):
        if not isinstance(v, dict) or not v:
            raise ValueError("El campo 'features' debe ser un objeto con pares {feature: valor}.")
        # Tipos numéricos
        for k, val in v.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"El valor de '{k}' debe ser numérico.")
        return v


class PredictResponse(BaseModel):
    predicted_class: int          # 0=Benigno, 1=Maligno
    prob_maligno: float           # Probabilidad de la clase 1
    label: str                    # "B" o "M"
    threshold_used: float
    missing_features: List[str] = []
    extra_features: List[str] = []


# ==============================
# Helpers
# ==============================
def ensure_loaded():
    if MODEL is None:
        raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {LOAD_ERROR}")

def order_and_validate(features: Dict[str, float]) -> (pd.DataFrame, List[str], List[str]):
    """
    Verifica que las claves coincidan EXACTAMENTE con FEATURE_NAMES y
    devuelve un DataFrame de una fila en el orden correcto.
    """
    if not FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="feature_names no disponible en el modelo.")

    expected = set(FEATURE_NAMES)
    got = set(features.keys())
    missing = sorted(list(expected - got))
    extra = sorted(list(got - expected))

    if missing or extra:
        # No rechazamos automáticamente: informamos qué falta/sobra para facilitar debugging,
        # pero retornamos 422 si hay diferencias de claves.
        raise HTTPException(
            status_code=422,
            detail={"msg": "Las claves de 'features' no coinciden con las esperadas.",
                    "missing_features": missing, "extra_features": extra,
                    "expected": FEATURE_NAMES}
        )

    # Ordenar según FEATURE_NAMES y chequear finitos
    row = [float(features[name]) for name in FEATURE_NAMES]
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    if not np.isfinite(df.to_numpy()).all():
        raise HTTPException(status_code=422, detail="Los valores deben ser finitos (no NaN/Inf).")
    return df, missing, extra

def do_predict(df: pd.DataFrame, confidence: float) -> Dict:
    proba = float(MODEL.predict_proba(df)[:, 1][0])   # prob de clase 1 (Maligno)
    pred = int(proba >= confidence)
    return {
        "predicted_class": pred,
        "prob_maligno": proba,
        "label": INV_MAP.get(pred, str(pred)),
        "threshold_used": float(confidence),
    }

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
    """
    Realiza la predicción para un único caso.
    - body.features: dict {feature: valor} con TODAS las features esperadas
    - confidence: umbral para clasificar como Maligno (1)
    """
    ensure_loaded()
    df, missing, extra = order_and_validate(body.features)
    result = do_predict(df, confidence)
    return PredictResponse(**result, missing_features=missing, extra_features=extra)

# Arranque local
if __name__ == "__main__":
    # uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
