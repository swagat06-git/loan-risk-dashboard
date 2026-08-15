"""
Data loading and cleaning utilities for the loan risk dashboard.
"""
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw loan data from a CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types and handle missing values.

    Written to tolerate the messiness of the real LendingClub export
    (percent signs as strings, "36 months" style term fields, etc.)
    as well as clean sample data.
    """
    df = df.copy()

    # Percent-as-string columns, e.g. "13.5%" -> 13.5
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.replace("%", "", regex=False).astype(float)
            )

    # Term as integer months, e.g. " 36 months" -> 36
    if "term" in df.columns and df["term"].dtype == object:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # Employment length as integer years, e.g. "10+ years" -> 10
    if "emp_length" in df.columns and df["emp_length"].dtype == object:
        df["emp_length"] = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float)

    if "loan_status" in df.columns:
        df = df.dropna(subset=["loan_status"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    df = df[df["loan_status"].isin(["Default","Fully Paid","Charged Off"])]
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary `default` column from the loan_status text field."""
    df = df.copy()
    bad_statuses = {"charged off", "default", "late (31-120 days)", "late (16-30 days)"}
    df["default"] = (
        df["loan_status"].astype(str).str.lower().isin(bad_statuses).astype(int)
    )
    return df
