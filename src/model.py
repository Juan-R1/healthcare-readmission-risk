"""Train a baseline model to predict 30‑day hospital readmission.

This script loads the processed dataset created by `src/data.py`, splits it into
training and test sets, trains a logistic regression model, evaluates it, and
prints the resulting metrics. Running the script requires scikit‑learn and
pandas installed.

You can execute this script from the project root via:

```
python src/model.py
```
"""

import pathlib
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "processed_diabetic_data.csv"
)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and target from the processed CSV."""
    df = pd.read_csv(DATA_FILE)
    target = df["readmitted_flag"]
    features = df.drop(columns=["readmitted_flag", "readmitted"])
    return features, target


def build_model(feature_df: pd.DataFrame) -> Pipeline:
    """Construct a preprocessing and modelling pipeline."""
    # Identify numeric and categorical columns
    numeric_cols = feature_df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    model = LogisticRegression(max_iter=200, n_jobs=-1)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return clf


def main() -> None:
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset contains {X.shape[0]} rows and {X.shape[1]} columns.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = build_model(X)
    print("Training model...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
