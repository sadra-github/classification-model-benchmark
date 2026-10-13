"""Unit tests for the classification benchmark."""

import tempfile
import unittest
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from src.classification_benchmark import evaluate_models, load_data, save_outputs


class ClassificationBenchmarkTests(unittest.TestCase):
    def test_load_data_shapes(self):
        X, y, target_names = load_data()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(target_names), 2)
        self.assertGreater(len(X), 100)

    def test_evaluation_contains_expected_models_and_metrics(self):
        X, y, _ = load_data()
        metrics, predictions = evaluate_models(X, y)
        self.assertEqual(set(metrics["model"]), {
            "DummyClassifier", "LogisticRegression", "RandomForest"
        })
        self.assertTrue({
            "accuracy", "precision_macro", "recall_macro", "f1_macro"
        }.issubset(metrics.columns))
        self.assertEqual(len(predictions), 3)
        self.assertTrue(metrics["f1_macro"].between(0, 1).all())

    def test_evaluation_is_reproducible(self):
        X, y, _ = load_data()
        first, _ = evaluate_models(X, y)
        second, _ = evaluate_models(X, y)
        self.assertEqual(first["model"].tolist(), second["model"].tolist())
        for column in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            self.assertTrue((first[column].round(10) == second[column].round(10)).all())

    def test_outputs_are_created(self):
        X, y, _ = load_data()
        metrics, predictions = evaluate_models(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            best_model = save_outputs(metrics, predictions, tmp)
            self.assertIn(best_model, set(metrics["model"]))
            self.assertTrue((Path(tmp) / "classification_metrics.csv").is_file())
            self.assertTrue((Path(tmp) / "confusion_matrix.png").is_file())


if __name__ == "__main__":
    unittest.main()
