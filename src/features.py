"""
Feature engineering for the loan risk model.
"""
import numpy as np
import pandas as pd

STATE_TO_REGION = {
    "CT":"Northeast","ME":"Northeast","MA":"Northeast","NH":"Northeast","RI":"Northeast",
    "VT":"Northeast","NJ":"Northeast","NY":"Northeast","PA":"Northeast",
    "IL":"Midwest","IN":"Midwest","MI":"Midwest","OH":"Midwest","WI":"Midwest",
    "IA":"Midwest","KS":"Midwest","MN":"Midwest","MO":"Midwest","NE":"Midwest",
    "ND":"Midwest","SD":"Midwest",
    "DE":"South","FL":"South","GA":"South","MD":"South","NC":"South","SC":"South",
    "VA":"South","DC":"South","WV":"South","AL":"South","KY":"South","MS":"South",
    "TN":"South","AR":"South","LA":"South","OK":"South","TX":"South",
    "AZ":"West","CO":"West","ID":"West","MT":"West","NV":"West","NM":"West",
    "UT":"West","WY":"West","AK":"West","CA":"West","HI":"West","OR":"West","WA":"West",
}


NUMERIC_FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "loan_to_income", "high_dti", "grade_rank", "emp_length",
]

CATEGORICAL_FEATURES = ["home_ownership", "purpose", "region"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Loan-to-income ratio
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_to_income"].fillna(df["loan_to_income"].median())

    # Income bands, handy for groupby segmentation in the dashboard
    df["income_band"] = pd.cut(
        df["annual_inc"],
        bins=[0, 30000, 60000, 100000, 150000, np.inf],
        labels=["<30k", "30-60k", "60-100k", "100-150k", "150k+"],
    )

    # High debt burden flag
    df["high_dti"] = (df["dti"] > 25).astype(int)

    # Numeric risk tier from letter grade (A best -> G worst)
    grade_order = {g: i for i, g in enumerate("ABCDEFG")}
    df["grade_rank"] = df["grade"].map(grade_order).fillna(3)
    df["region"] = df["addr_state"].map(STATE_TO_REGION).fillna("unknown")

    return df
