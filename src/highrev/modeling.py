"""Exercise 2: separate modeling responsibilities without changing the baseline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

NUMERIC = ["pages_visited", "age_first_order"]
CATEGORICAL = ["gender", "campaign", "ip_address_country", "browser", "os"]
TARGET = "high_revenue"
FEATURES = [*NUMERIC, *CATEGORICAL]
BASELINE_CLASSIFIER_ARGS = {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}


def _as_str(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve baseline categorical normalization; keep this helper importable."""
    normalized = frame.astype("string").fillna("missing")
    return normalized.astype(object)


def build_model(model_options: dict | None = None) -> Pipeline:
    """Return the unfitted baseline preprocessing and classifier pipeline."""
    clf_kwargs = dict(BASELINE_CLASSIFIER_ARGS)
    if model_options:
        clf_kwargs.update(model_options)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("as_str", FunctionTransformer(_as_str, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", transformer),
            ("classifier", LogisticRegression(**clf_kwargs)),
        ]
    )


def split_data(
    labeled: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test frames with declared features and TARGET, stratified by TARGET."""
    subset = labeled.loc[:, [*FEATURES, TARGET]]
    train_df, test_df = train_test_split(
        subset,
        test_size=test_size,
        random_state=random_state,
        stratify=subset[TARGET],
    )
    return train_df, test_df


def train_model(train: pd.DataFrame, model_options: dict | None = None) -> Pipeline:
    """Build and fit a model on train only; do not mutate train."""
    pipeline = build_model(model_options=model_options)
    X_train = train.loc[:, FEATURES]
    y_train = train[TARGET].astype(int)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    """Return roc_auc, accuracy, f1 using the baseline probability and decision rules."""
    X_eval = test.loc[:, FEATURES]
    y_true = test[TARGET].astype(int)
    pos_scores = model.predict_proba(X_eval)[:, 1]
    hard_preds = (pos_scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, pos_scores)),
        "accuracy": float(accuracy_score(y_true, hard_preds)),
        "f1": float(f1_score(y_true, hard_preds)),
    }