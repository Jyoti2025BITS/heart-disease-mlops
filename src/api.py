import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("heart_disease_api")


MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "heart_disease_pipeline.joblib"
)


model = joblib.load(MODEL_PATH)


REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)


REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)


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


@app.middleware("http")
async def observe_requests(request: Request, call_next):

    start = time.perf_counter()

    status = 500

    try:
        response = await call_next(request)

        status = response.status_code

        return response

    finally:
        duration = time.perf_counter() - start

        endpoint = request.url.path

        REQUEST_COUNT.labels(
            request.method,
            endpoint,
            str(status),
        ).inc()

        REQUEST_LATENCY.labels(
            request.method,
            endpoint,
        ).observe(duration)

        logger.info(
            "method=%s path=%s status=%s duration_seconds=%.4f",
            request.method,
            endpoint,
            status,
            duration,
        )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        headers={"Cache-Control": "no-store"},
        media_type=CONTENT_TYPE_LATEST,
    )


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
                float(model.predict_proba(features)[0][1]),
                4,
            )

        return result

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API logs.",
        ) from exc
