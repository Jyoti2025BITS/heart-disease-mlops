import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/raw/processed.cleveland.data")

MODEL_DIR = Path("models")

REPORT_DIR = Path("reports")


COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def build_pipeline(model):

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}. Run python src/download_data.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")

    y = (df["num"] > 0).astype(int)

    X = df.drop(columns=["num"])

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    tracking_db = Path("mlflow.db").resolve()

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    mlflow.set_experiment("heart-disease-classification")

    results = {}

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)

        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, error_score="raise"
        )

        metrics = {
            metric: round(float(np.mean(scores[f"test_{metric}"])), 4)
            for metric in scoring
        }

        results[name] = metrics

        pipeline.fit(X, y)

        artifact = MODEL_DIR / f"{name}.joblib"

        joblib.dump(pipeline, artifact)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)

            mlflow.log_param("cv_folds", 5)

            mlflow.log_param("random_state", 42)

            mlflow.log_param("artifact_format", "joblib")

            mlflow.log_metrics(metrics)

            mlflow.log_artifact(str(artifact), artifact_path="joblib")

        print(f"{name}: {metrics}")

    best_name = max(results, key=lambda name: results[name]["roc_auc"])

    best_model = build_pipeline(models[best_name])

    best_model.fit(X, y)

    final_path = MODEL_DIR / "heart_disease_pipeline.joblib"

    joblib.dump(best_model, final_path)

    with open(REPORT_DIR / "model_comparison.json", "w") as f:
        json.dump(
            {
                "cross_validation": results,
                "selected_model": best_name,
                "selection_metric": "roc_auc",
            },
            f,
            indent=2,
        )

    print(f"\nSelected model: {best_name}")

    print(f"Saved model: {final_path}")

    print(f"MLflow database: {tracking_db}")


if __name__ == "__main__":
    main()
