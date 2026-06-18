"""Generate portfolio visuals for the readmission-risk baseline model."""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from model import build_model, load_data


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed_diabetic_data.csv"


def save_readmission_distribution() -> None:
    """Save the 30-day readmission target distribution chart."""
    df = pd.read_csv(PROCESSED_DATA, low_memory=False)
    counts = (
        df["readmitted_flag"]
        .map({0: "Not readmitted within 30 days", 1: "Readmitted within 30 days"})
        .value_counts()
        .reindex(["Not readmitted within 30 days", "Readmitted within 30 days"])
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", legend=False)
    ax.set_title("30-Day Readmission Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Encounter count")
    ax.bar_label(ax.containers[0], fmt="%d")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "readmission_distribution.png", dpi=160)
    plt.close()


def save_model_visuals() -> None:
    """Train the baseline model and save evaluation visuals."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_model(X)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds,
        display_labels=["Not <30", "<30"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title("Baseline Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "readmission_confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probas)
    plt.title("Baseline Logistic Regression ROC Curve")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "readmission_roc_curve.png", dpi=160)
    plt.close()

    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    coefficients = clf.named_steps["model"].coef_[0]
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(15)
        .sort_values("coefficient")
    )

    plt.figure(figsize=(9, 6))
    colors = ["#c0392b" if value < 0 else "#2c7fb8" for value in coef_df["coefficient"]]
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.title("Largest Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "readmission_feature_coefficients.png", dpi=160)
    plt.close()


def main() -> None:
    IMAGES_DIR.mkdir(exist_ok=True)
    save_readmission_distribution()
    save_model_visuals()
    print(f"Saved visuals to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
