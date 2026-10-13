"""Compare baseline classification models on a reproducible dataset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def load_data():
    """Return features and labels from a built-in scikit-learn dataset."""
    dataset = load_breast_cancer()
    return dataset.data, dataset.target, dataset.target_names


def build_models():
    """Create fresh model instances for a fair, reproducible comparison."""
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def evaluate_models(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Fit models on a stratified split and return metrics plus test predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rows = []
    predictions = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = (y_test, y_pred)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(
        "f1_macro", ascending=False
    ).reset_index(drop=True)
    return metrics, predictions


def save_outputs(metrics, predictions, output_dir=OUTPUT_DIR):
    """Save metrics and a confusion matrix for the top macro-F1 model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "classification_metrics.csv", index=False)

    best_model = metrics.iloc[0]["model"]
    y_test, y_pred = predictions[best_model]
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=load_data()[2],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(f"Confusion Matrix: {best_model}")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160)
    return best_model


def main():
    X, y, _ = load_data()
    metrics, predictions = evaluate_models(X, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model = save_outputs(metrics, predictions)
    print("Classification benchmark results (sorted by macro F1):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Best model by macro F1: {best_model}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'classification_metrics.csv'}")
    print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
