from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "heart_disease_pipeline.joblib"
)


model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from clinical features.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)

    sex: int = Field(..., ge=0, le=1)

    cp: int = Field(..., ge=1, le=4)

    trestbps: float = Field(..., gt=0, le=300)

    chol: float = Field(..., ge=0, le=700)

    fbs: int = Field(..., ge=0, le=1)

    restecg: int = Field(..., ge=0, le=2)

    thalach: float = Field(..., gt=0, le=300)

    exang: int = Field(..., ge=0, le=1)

    oldpeak: float = Field(..., ge=0, le=10)

    slope: int = Field(..., ge=1, le=3)

    ca: float = Field(..., ge=0, le=4)

    thal: float = Field(..., ge=3, le=7)


@app.get("/health")
def health():

    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientData):

    try:
        features = pd.DataFrame([patient.model_dump()])

        prediction = int(model.predict(features)[0])

        result = {
            "prediction": prediction,
            "interpretation": (
                "heart disease detected by model"
                if prediction == 1
                else "heart disease not detected by model"
            ),
        }

        if hasattr(model, "predict_proba"):
            result["probability_class_1"] = round(
                float(model.predict_proba(features)[0][1]), 4
            )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from exc
