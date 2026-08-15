# Loan Default & Credit Risk Dashboard

I built this to figure out which loans are actually risky, using LendingClub's public dataset of ~2 million real loans issued between 2007 and 2018. It's not a notebook full of charts — it's a live tool: a portfolio-level view of where default risk concentrates, plus a form where you can plug in an applicant's numbers and get a risk score back.

**Live app:** https://loan-risk-dashboard-sp.streamlit.app

## Why this project

I wanted something closer to what a data analyst actually does day to day — not just fitting a model and reporting an accuracy number, but asking "so what should the business actually do with this." That meant spending more time on the segmentation and the dashboard than on squeezing out an extra point of AUC.

## What's in it

- **Portfolio Overview** — default rate broken down by loan grade, region, and purpose, plus a DTI-vs-interest-rate scatter to see where risk clusters
- **Applicant Risk Check** — enter an applicant's details, get a predicted default probability and a low/moderate/high flag
- A logistic regression model underneath (chosen over random forest deliberately — more on that below)

## Findings

- Default rate climbs steadily from grade A (~5-6%) up to F/G (~35-40%+) — confirms LendingClub's own grading system is doing real work as a risk signal.
- Region barely moves the needle — Midwest, Northeast, South, West are all clustered around 15-20%, nowhere near the spread you see across grades. Worth saying explicitly: grade matters far more than geography.

## A decision I made on purpose: logistic regression over random forest

I tried both. Random forest edged it out slightly on raw AUC, but I went with logistic regression for the deployed version anyway — the coefficients tell you *why* a given loan looks risky (high DTI, low income relative to loan size, etc.), which matters more for a tool meant to be used by a person making a lending decision than a marginal accuracy gain would. If you're the type to disagree with that tradeoff, the random forest path is still in `train_model.py --model random_forest` — worth running both and forming your own opinion on it.

## Model performance

Logistic regression, trained on ~1.3M resolved loans (Fully Paid, Charged Off, or Default — I excluded loans still "Current" or "In Grace Period," since we don't actually know how those end yet):

- AUC: 0.704
- Recall on defaults: 0.62 (catches ~6 in 10 actual defaults)
- Precision on defaults: 0.32 (roughly 1 in 3 flagged loans actually defaults)

That precision number is the honest tradeoff here — `class_weight="balanced"` pushes the model to catch more real defaults, at the cost of more false alarms on safe borrowers. For a lending business, that's usually the right side to err on, but it's worth knowing you're trading one kind of mistake for another, not eliminating mistakes.

## Running it locally

```bash
pip install -r requirements.txt
python generate_sample_data.py   # optional: quick synthetic data to test with
python train_model.py --data data/your_data.csv
streamlit run app.py
```

The real dataset (`accepted_2007_to_2018Q4.csv` from [LendingClub on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)) isn't included in this repo — it's too big for GitHub. Download it separately and point `--data` at it.

## What I'd do with more time

- Pull in `verification_status` and `earliest_cr_line` (credit history length) as features — I left these out to keep the first version manageable, but they're plausible signal
- Calibrate the predicted probabilities properly (right now they're raw model outputs, not adjusted to match real-world default frequencies)
- Swap the CSV read for a proper SQL layer — the groupby aggregations in the dashboard would translate directly to `GROUP BY` queries
