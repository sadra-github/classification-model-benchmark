# Classification Model Benchmark

A small, reproducible Python project that compares baseline classification models on a synthetic dataset.

## Models
- `DummyClassifier` (majority-class baseline)
- `LogisticRegression`
- `RandomForestClassifier`

## Evaluation
The script reports accuracy, macro precision, macro recall, and macro F1-score, and saves a confusion-matrix plot for the best model by macro F1-score.

## Dataset and protocol
- Uses scikit-learn's built-in breast cancer dataset.
- Uses a stratified train/test split (80%/20%) with a fixed random seed.
- The dataset is included with scikit-learn; no external download or private data is required.
- This is a demonstration benchmark, not a claim that one model is universally best.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Run

```bash
python src/classification_benchmark.py
```

The script creates:
- `outputs/classification_metrics.csv`
- `outputs/confusion_matrix.png`

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
classification-model-benchmark/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   └── classification_benchmark.py
├── tests/
│   └── test_classification.py
└── outputs/
    └── .gitkeep
```

## License
MIT
