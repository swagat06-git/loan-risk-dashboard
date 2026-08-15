"""
Loan Default & Credit Risk Dashboard

Run with:
    streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model import load_pipeline

st.set_page_config(page_title="Loan Risk Dashboard", layout="wide")


@st.cache_data
def get_data():
    return pd.read_csv("data/processed_loans.csv")


@st.cache_resource
def get_model():
    return load_pipeline()


df = get_data()
model = get_model()

st.title("💳 Loan Default & Credit Risk Dashboard")

tab1, tab2 = st.tabs(["📊 Portfolio Overview", "🔍 Applicant Risk Check"])

# ---------------------------------------------------------------- Overview
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loans", f"{len(df):,}")
    col2.metric("Overall Default Rate", f"{df['default'].mean() * 100:.1f}%")
    col3.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")
    col4.metric("Avg Interest Rate", f"{df['int_rate'].mean():.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        by_grade = df.groupby("grade")["default"].mean().reset_index()
        fig = px.bar(by_grade, x="grade", y="default",
                     title="Default Rate by Loan Grade",
                     labels={"default": "Default Rate"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_region = df.groupby("region")["default"].mean().reset_index()
        fig = px.bar(by_region, x="region", y="default",
                     title="Default Rate by Region",
                     labels={"default": "Default Rate"})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        by_purpose = (
            df.groupby("purpose")["default"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(by_purpose, x="purpose", y="default", title="Default Rate by Loan Purpose")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sample = df.sample(min(1000, len(df)), random_state=0)
        fig = px.scatter(sample, x="dti", y="int_rate", color="default",
                          title="DTI vs Interest Rate (colored by default)", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------- Applicant check
with tab2:
    st.subheader("Score a new applicant")
    c1, c2, c3 = st.columns(3)

    with c1:
        loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 10000, step=500)
        term = st.selectbox("Term (months)", [36, 60])
        int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 12.0)
        installment = loan_amnt / term * (1 + int_rate / 100)
        emp_length = st.slider("Years employed", 0, 10, 3)

    with c2:
        annual_inc = st.number_input("Annual income ($)", 10000, 300000, 55000, step=1000)
        dti = st.slider("Debt-to-income ratio", 0.0, 40.0, 18.0)
        home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE"])
        purpose = st.selectbox("Loan purpose", sorted(df["purpose"].unique()))

    with c3:
        region = st.selectbox("Region", sorted(df["region"].unique()))
        delinq_2yrs = st.slider("Delinquencies (2yr)", 0, 5, 0)
        open_acc = st.slider("Open accounts", 1, 30, 10)
        pub_rec = st.slider("Public records", 0, 3, 0)
        revol_bal = st.number_input("Revolving balance ($)", 0, 100000, 8000, step=500)
        revol_util = st.slider("Revolving utilization (%)", 0.0, 100.0, 40.0)
        total_acc = open_acc + 5

    grade_order = {g: i for i, g in enumerate("ABCDEFG")}
    # crude grade estimate from the rate, just to feed grade_rank in the single-applicant form
    if int_rate < 8:
        grade_guess = "A"
    elif int_rate < 12:
        grade_guess = "B"
    elif int_rate < 16:
        grade_guess = "C"
    elif int_rate < 20:
        grade_guess = "D"
    elif int_rate < 24:
        grade_guess = "E"
    else:
        grade_guess = "F"

    row = pd.DataFrame([{
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "annual_inc": annual_inc, "dti": dti,
        "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
        "loan_to_income": loan_amnt / max(annual_inc, 1),
        "high_dti": int(dti > 25),
        "grade_rank": grade_order[grade_guess],
        "emp_length": emp_length,
        "home_ownership": home_ownership, "purpose": purpose, "region": region,
    }])

    if st.button("Check risk", type="primary"):
        prob = model.predict_proba(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0, 1]
        st.metric("Predicted default probability", f"{prob * 100:.1f}%")
        if prob < 0.10:
            st.success("Low risk")
        elif prob < 0.25:
            st.warning("Moderate risk")
        else:
            st.error("High risk")
