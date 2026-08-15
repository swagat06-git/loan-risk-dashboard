"""
Trains the loan default risk model end-to-end and saves the fitted pipeline
plus a processed copy of the data for the dashboard to read.

Usage:
    python train_model.py --data data/sample_loans.csv --model logistic
"""
import argparse

from src.data_prep import add_target, clean_data, load_data
from src.features import engineer_features
from src.model import save_pipeline, train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_loans.csv")
    parser.add_argument("--model", default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--out", default="models/risk_model.joblib")
    args = parser.parse_args()

    df = load_data(args.data)
    df = clean_data(df)
    df = add_target(df)
    df = engineer_features(df)

    pipeline, metrics = train_and_evaluate(df, model_type=args.model)

    print(f"AUC: {metrics['auc']:.3f}")
    print(metrics["report"])

    save_pipeline(pipeline, args.out)
    print(f"Saved model to {args.out}")

    df.to_csv("data/processed_loans.csv", index=False)
    print("Saved processed data to data/processed_loans.csv")


if __name__ == "__main__":
    main()
