# Loan Default & Credit Risk Dashboard

A deployable analyst tool: explore portfolio-level default trends and score
individual loan applicants for risk. Built with pandas, scikit-learn, and
Streamlit.

## What this demonstrates
- End-to-end pipeline: raw data -> cleaning -> feature engineering -> model -> dashboard
- Business-facing segmentation (default rate by grade, region, purpose, income band)
- An interpretable model (logistic regression) with an option to compare against random forest
- A deployed, interactive tool rather than a static notebook

## Project structure
```
loan-risk-dashboard/
├── app.py                   # Streamlit dashboard (run this)
├── train_model.py           # Trains and saves the model
├── generate_sample_data.py  # Creates synthetic data to run immediately
├── requirements.txt
├── src/
│   ├── data_prep.py         # load_data, clean_data, add_target
│   ├── features.py          # engineer_features, feature lists
│   └── model.py             # build_pipeline, train_and_evaluate, save/load
├── data/                    # sample_loans.csv, processed_loans.csv
└── models/                  # risk_model.joblib (saved after training)
```

## Quickstart (with synthetic data)
```bash
pip install -r requirements.txt
python generate_sample_data.py     # creates data/sample_loans.csv
python train_model.py              # trains model, writes models/ and processed data
streamlit run app.py               # opens the dashboard at localhost:8501
```

## Using the real LendingClub data (recommended before you ship this)
The synthetic data is just a stand-in so the pipeline runs on day one — for
your actual portfolio project, swap it for real data:

1. Download the LendingClub loan data, e.g. from
   https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Save it as `data/sample_loans.csv` (or point `--data` at wherever you put it):
   ```bash
   python train_model.py --data data/lending_club_2018.csv
   ```
3. Re-run `streamlit run app.py` — the dashboard reads whatever `train_model.py`
   last processed.

Note: the real file has different/extra columns than the synthetic generator.
Check `src/data_prep.py` and `src/features.py` and adjust column names as needed —
working through that mismatch is itself good practice for a real analyst project.

## Trying the random forest model
```bash
python train_model.py --model random_forest
```
Compare the printed AUC against the logistic regression baseline — this is a
good thing to mention in your case-study writeup (interpretability vs. performance
tradeoff).

## Deploying
The fastest free option given this stack is **Streamlit Community Cloud**:
1. Push this folder to a public GitHub repo (include `data/processed_loans.csv`
   and `models/risk_model.joblib` after training, or add a build step that runs
   `generate_sample_data.py` + `train_model.py` on startup).
2. Go to https://share.streamlit.io, connect your GitHub repo, point it at `app.py`.
3. It builds automatically from `requirements.txt`.

## Next steps to strengthen this for applications
- Swap in the real LendingClub dataset (see above)
- Add a SQL layer: load the data into SQLite/Postgres and query it instead of
  reading a CSV directly — analyst interviews test SQL heavily
- Add a short case-study write-up: the business question, what you found,
  what you'd recommend a lending team do differently
- Add confidence/calibration info to the risk score, not just a raw probability
