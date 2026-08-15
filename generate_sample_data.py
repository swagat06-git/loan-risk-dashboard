"""
Generates a synthetic loan dataset shaped like the real LendingClub export,
so we can run the whole pipeline immediately without downloading anything.

We will swap this out for real data once we're ready to make this portfolio-worthy:

We will just point at the real CSV instead.

Usage:
    python generate_sample_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)  #for reproducibility


def generate_sample_loans(n: int = 5000) -> pd.DataFrame:
    grade_probs = [0.18, 0.22, 0.22, 0.16, 0.12, 0.06, 0.04]
    grades = np.random.choice(list("ABCDEFG"), size=n, p=grade_probs)
    grade_risk = {"A": 0.03, "B": 0.07, "C": 0.12, "D": 0.18, "E": 0.25, "F": 0.33, "G": 0.42}

    annual_inc = np.round(np.random.lognormal(mean=10.9, sigma=0.5, size=n), -2)
    loan_amnt = np.round(np.random.uniform(1000, 40000, size=n), -2)
    term = np.random.choice([36, 60], size=n, p=[0.7, 0.3])
    int_rate = np.round(np.random.uniform(5, 30, size=n), 2)
    installment = np.round(loan_amnt / term * (1 + int_rate / 100), 2)
    dti = np.round(np.random.uniform(0, 40, size=n), 2)
    emp_length = np.random.randint(0, 11, size=n)
    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE"], size=n, p=[0.4, 0.15, 0.45])
    purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "other"],
        size=n, p=[0.45, 0.2, 0.1, 0.08, 0.07, 0.1],
    )
    region = np.random.choice(["Northeast", "Midwest", "South", "West"], size=n)
    delinq_2yrs = np.random.poisson(0.3, size=n)
    open_acc = np.random.randint(2, 25, size=n)
    pub_rec = np.random.poisson(0.1, size=n)
    revol_bal = np.round(np.random.uniform(0, 50000, size=n), 2)
    revol_util = np.round(np.random.uniform(0, 100, size=n), 2)
    total_acc = open_acc + np.random.randint(0, 15, size=n)

    base_risk = np.array([grade_risk[g] for g in grades])
    risk_score = (
        base_risk
        + (dti > 25) * 0.05
        + (annual_inc < 30000) * 0.05
        + np.random.normal(0, 0.05, size=n)
    ).clip(0.01, 0.9)
    default = np.random.binomial(1, risk_score)
    loan_status = np.where(default == 1, "Charged Off", "Fully Paid")

    return pd.DataFrame({
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "grade": grades,
        "sub_grade": [f"{g}{np.random.randint(1, 6)}" for g in grades],
        "emp_length": emp_length, "home_ownership": home_ownership,
        "annual_inc": annual_inc, "purpose": purpose, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
        "loan_status": loan_status, "region": region,
    })


if __name__ == "__main__":
    df = generate_sample_loans()
    df.to_csv("data/sample_loans.csv", index=False)
    print(f"Wrote {len(df)} rows to data/sample_loans.csv")
