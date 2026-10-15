"""Train the Iris classification pipeline used by the demo API.

This is an explicit, standalone step. The FastAPI application never trains
or retrains a model itself -- it only loads the artifacts this script writes.

Usage (from repo root, PowerShell):
    .venv\\Scripts\\python.exe scripts\\train.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.model import FEATURE_ORDER, METADATA_FILENAME, MODEL_FILENAME, get_artifacts_dir  # noqa: E402

MODEL_VERSION = "iris-logreg-1.0.0"
RANDOM_SEED = 42
TEST_SIZE = 0.2


def train(output_dir: Path) -> dict:
    iris = load_iris()
    # iris.feature_names are in cm already; load_iris's column order is exactly
    # [sepal length, sepal width, petal length, petal width], matching
    # FEATURE_ORDER in app/model.py.
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # LogisticRegression.classes_ holds the integer labels in the order
    # probabilities are returned. For load_iris, labels 0/1/2 map 1:1 onto
    # iris.target_names in that same order, so this list is the class order
    # the API must use when labeling predict_proba output.
    classes_ = pipeline.named_steps["classifier"].classes_
    target_classes = [str(iris.target_names[c]) for c in classes_]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "iris (sklearn.datasets.load_iris)",
            "n_samples_total": int(X.shape[0]),
            "n_samples_train": int(X_train.shape[0]),
            "n_samples_test": int(X_test.shape[0]),
        },
        "feature_order": FEATURE_ORDER,
        "feature_units": "centimeters",
        "target_classes": target_classes,
        "held_out_accuracy": accuracy,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write model.joblib and metadata.json into. "
        "Defaults to <repo_root>/artifacts.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else get_artifacts_dir()
    metadata = train(output_dir)

    print(f"Trained {metadata['model_version']}")
    print(f"Held-out accuracy: {metadata['held_out_accuracy']:.4f}")
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
