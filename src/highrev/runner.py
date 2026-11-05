"""Exercise 2: put file access and orchestration behind an explicit function call."""

import pickle
from pathlib import Path

import pandas as pd

from highrev.data_processing import label_customers, prepare_customers, prepare_orders
from highrev.modeling import evaluate_model, split_data, train_model

RAW_DATA_DIR = Path("data/01_raw")
MODEL_ARTIFACT = Path("model.pickle")


def main() -> None:
    """Run from the exercise project root.

    Read data/01_raw/customers.csv and orders.csv with string customer IDs.
    Call preparation, labeling, splitting, training, and evaluation functions.
    Print all three metrics and write model.pickle in the project root.
    Importing any module must not call this function or cause these side effects.
    """
    raw_customers = pd.read_csv(RAW_DATA_DIR / "customers.csv", dtype={"customerID": str})
    raw_orders = pd.read_csv(RAW_DATA_DIR / "orders.csv", dtype={"customer_id": str})

    customer_orders = prepare_orders(raw_orders)
    prepared_customers = prepare_customers(raw_customers, customer_orders)
    labeled_customers = label_customers(prepared_customers)

    train_split, test_split = split_data(labeled_customers)
    fitted_pipeline = train_model(train_split)
    scores = evaluate_model(fitted_pipeline, test_split)

    for metric_name, label in (("roc_auc", "ROC AUC "), ("accuracy", "accuracy"), ("f1", "f1      ")):
        print(f"{label} {scores[metric_name]:.3f}")

    MODEL_ARTIFACT.write_bytes(pickle.dumps(fitted_pipeline))
    print(f"saved {MODEL_ARTIFACT.name}")