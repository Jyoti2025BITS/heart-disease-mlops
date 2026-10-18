from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_prediction_endpoint():

    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]

    assert 0.0 <= result["probability_class_1"] <= 1.0

    assert "interpretation" in result


def test_invalid_input_rejected():

    response = client.post("/predict", json={"age": -1})

    assert response.status_code == 422
