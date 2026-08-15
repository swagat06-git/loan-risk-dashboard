"""
Model training, evaluation, and persistence.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(model_type: str = "logistic") -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "logistic":
        # Interpretable coefficients matter for an analyst-facing tool
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


def train_and_evaluate(df: pd.DataFrame, model_type: str = "logistic"):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, pipeline.predict(X_test))

    return pipeline, {"auc": auc, "report": report}


def save_pipeline(pipeline, path: str = "models/risk_model.joblib") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str = "models/risk_model.joblib"):
    return joblib.load(path)
