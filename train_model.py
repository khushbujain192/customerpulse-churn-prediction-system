from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DEFAULT_DATASET_PATH = DATA_DIR / "customer_churn.csv"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

FEATURE_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "online_security",
    "online_backup",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "payment_method",
    "contract_type",
    "paperless_billing",
    "monthly_charges",
    "total_charges",
]

TARGET_COLUMN = "churn"


@dataclass
class TrainingResult:
    model_path: Path
    metrics_path: Path
    accuracy: float
    roc_auc: float


def _generate_synthetic_dataset(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], size=rows)
    senior_citizen = rng.choice([0, 1], p=[0.83, 0.17], size=rows)
    partner = rng.choice(["Yes", "No"], p=[0.52, 0.48], size=rows)
    dependents = rng.choice(["Yes", "No"], p=[0.31, 0.69], size=rows)
    tenure = rng.integers(1, 73, size=rows)

    online_security = rng.choice(["Yes", "No"], p=[0.42, 0.58], size=rows)
    online_backup = rng.choice(["Yes", "No"], p=[0.43, 0.57], size=rows)
    tech_support = rng.choice(["Yes", "No"], p=[0.39, 0.61], size=rows)
    streaming_tv = rng.choice(["Yes", "No"], p=[0.48, 0.52], size=rows)
    streaming_movies = rng.choice(["Yes", "No"], p=[0.47, 0.53], size=rows)

    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        p=[0.33, 0.23, 0.22, 0.22],
        size=rows,
    )

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        p=[0.58, 0.23, 0.19],
        size=rows,
    )

    paperless_billing = rng.choice(["Yes", "No"], p=[0.61, 0.39], size=rows)

    monthly_charges = np.clip(rng.normal(loc=70, scale=24, size=rows), 18, 130)
    total_charges = (monthly_charges * tenure) + rng.normal(loc=0, scale=45, size=rows)
    total_charges = np.clip(total_charges, 18, None)

    linear_score = (
        -1.6
        + 0.022 * monthly_charges
        - 0.024 * tenure
        + 0.85 * (contract_type == "Month-to-month")
        - 0.8 * (contract_type == "Two year")
        + 0.33 * (payment_method == "Electronic check")
        + 0.26 * (paperless_billing == "Yes")
        + 0.24 * (tech_support == "No")
        + 0.2 * (online_security == "No")
        + 0.19 * senior_citizen
        + rng.normal(0, 0.22, size=rows)
    )

    churn_probability = 1.0 / (1.0 + np.exp(-linear_score))
    churn = np.where(rng.uniform(size=rows) < churn_probability, "Yes", "No")

    df = pd.DataFrame(
        {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "online_security": online_security,
            "online_backup": online_backup,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "payment_method": payment_method,
            "contract_type": contract_type,
            "paperless_billing": paperless_billing,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "churn": churn,
        }
    )
    return df


def _load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _generate_synthetic_dataset()
    dataset.to_csv(dataset_path, index=False)
    return dataset


def _top_feature_importance(feature_names: List[str], coefficients: np.ndarray, n: int = 8) -> Dict[str, List[Dict[str, float]]]:
    indices_desc = np.argsort(coefficients)[::-1]
    indices_asc = np.argsort(coefficients)

    top_positive = [
        {"feature": feature_names[idx], "coefficient": float(coefficients[idx])}
        for idx in indices_desc[:n]
    ]
    top_negative = [
        {"feature": feature_names[idx], "coefficient": float(coefficients[idx])}
        for idx in indices_asc[:n]
    ]

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
    }


def train_and_save_model(dataset_path: Path = DEFAULT_DATASET_PATH) -> TrainingResult:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_or_create_dataset(dataset_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' column.")

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    if y.isna().any():
        raise ValueError("Target column 'churn' must contain only Yes/No values.")

    numeric_features = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]
    categorical_features = [feature for feature in FEATURE_COLUMNS if feature not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    feature_importance = _top_feature_importance(feature_names, coefficients)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "dataset_path": str(dataset_path),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classification_report": report,
        "confusion_matrix": matrix,
        "feature_importance": feature_importance,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return TrainingResult(
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        accuracy=accuracy,
        roc_auc=roc_auc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn model")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to CSV dataset",
    )

    args = parser.parse_args()
    result = train_and_save_model(Path(args.dataset))

    print("Model trained successfully.")
    print(f"Saved model: {result.model_path}")
    print(f"Saved metrics: {result.metrics_path}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"ROC-AUC: {result.roc_auc:.4f}")


if __name__ == "__main__":
    main()
