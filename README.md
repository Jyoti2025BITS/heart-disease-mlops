# Heart Disease Classification — MLOps Assignment

## Overview
This project builds a binary classification pipeline using the Cleveland heart-disease dataset. It includes exploratory data analysis (EDA), model training and comparison, MLflow experiment tracking, a FastAPI prediction endpoint, automated tests, and a GitHub Actions CI workflow.

> **Important:** This is an educational machine-learning project. Predictions are not medical diagnoses and must not be used to make healthcare decisions.

## Repository structure
```text
.
├── data/raw/processed.cleveland.data
├── models/
│   ├── heart_disease_pipeline.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── reports/model_comparison.json
├── screenshots/eda/
├── src/
│   ├── api.py
│   ├── download_data.py
│   ├── eda.py
│   ├── inspect_data.py
│   └── train.py
├── tests/test_api.py
├── requirements.txt
├── pytest.ini
└── .github/workflows/ci.yml
```

## Environment setup
Use Python 3.12 (the CI workflow uses Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project
Run commands from the repository root.

### Inspect / explore the data
```bash
python src/inspect_data.py
python src/eda.py
```

The EDA plots are saved under `screenshots/eda/`.

### Train and compare models
```bash
python src/train.py
```

Training logs model parameters, evaluation metrics, and model artifacts to MLflow. The tracking URI is configured in `src/train.py` to use the local SQLite database `mlflow.db`.

### View MLflow
If the database and experiment already exist, start the UI from the repository root:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Open the forwarded port 5000 in the Prayogshala VM environment, if port forwarding is supported.

### Run the prediction API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `/docs` on the same host and port. If port 8000 is already in use, an API process may already be running; check it before starting another one.

Health check:
```bash
curl http://localhost:8000/health
```

Example prediction request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":3}'
```

The API returns a model prediction and `probability_class_1`. The sample request is for demonstrating the endpoint only, not for clinical interpretation.

## Tests and code quality
```bash
ruff check src tests
ruff format --check src tests
python -m pytest -v
```

The GitHub Actions workflow runs dependency installation, Ruff linting, formatting validation, and the test suite on pushes and pull requests to `main`.

## Recorded model results
The current MLflow experiment `heart-disease-classification` contains two finished runs:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8349 | 0.9099 |
| Random Forest | 0.8349 | 0.9125 |

Random Forest has the higher ROC-AUC in these recorded runs, while both models have the same recorded accuracy. These metrics describe the evaluated dataset/split and do not establish clinical usefulness.

## Current verification status
- GitHub Actions CI: successful run observed.
- API health endpoint: returned `{"status":"healthy","model_loaded":true}`.
- API prediction endpoint: returned a valid JSON prediction response.
- MLflow: experiment found; two runs reported `FINISHED`.
- EDA: four plots are present under `screenshots/eda/`.

## Limitations and responsible use
- This is a learning and demonstration project, not a validated medical device.
- Model performance can vary with data splits, preprocessing, population, and deployment conditions.
- A prediction must not be interpreted as a diagnosis or a substitute for professional medical care.




## Docker Deployment



Build and run the API:



    docker build -t heart-disease-api:latest .

    docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest



Or use Docker Compose:



    docker compose up -d --build

    docker compose ps

    docker compose logs --tail=50



Test the health endpoint:



    curl http://localhost:8000/health



Test prediction:



    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":3}'



## Kubernetes Deployment



Kubernetes manifests are available in `deployment/`. Deployment requires a configured Kubernetes cluster, `kubectl`, and an image accessible to the cluster.



    kubectl apply -f deployment/deployment.yaml

    kubectl apply -f deployment/service.yaml



The manifests have been prepared but have not yet been deployed to a Kubernetes cluster.



## Disclaimer



This project is for educational purposes only. Predictions are not medical diagnoses.

