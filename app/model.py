"""Artifact loading helpers shared by the API and the test suite.

Paths are resolved relative to this file (or an explicit override), never
relative to the process's current working directory, so the API behaves the
same whether it is started from the repo root or from inside scripts/.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

# Order the model was trained on. The API must build feature vectors in
# exactly this order before calling pipeline.predict / predict_proba.
FEATURE_ORDER = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

TRAIN_COMMAND = ".venv\\Scripts\\python.exe scripts\\train.py"


def get_artifacts_dir() -> Path:
    """Resolve the artifacts directory.

    Honors the MODEL_ARTIFACTS_DIR environment variable (used by tests to
    point at an isolated temp directory) and otherwise falls back to
    <repo_root>/artifacts, computed from this file's own location.
    """
    override = os.environ.get("MODEL_ARTIFACTS_DIR")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts"


class ArtifactsNotFoundError(RuntimeError):
    """Raised when the trained pipeline/metadata cannot be found at startup."""


@dataclass
class LoadedModel:
    pipeline: Any
    metadata: dict


def load_model_artifacts(artifacts_dir: Path | None = None) -> LoadedModel:
    """Load the trained pipeline and its metadata from disk.

    Raises ArtifactsNotFoundError with the exact training command to run if
    either file is missing. Never trains or retrains a model itself.
    """
    directory = artifacts_dir if artifacts_dir is not None else get_artifacts_dir()
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    missing = [p.name for p in (model_path, metadata_path) if not p.exists()]
    if missing:
        raise ArtifactsNotFoundError(
            f"Missing model artifact(s) {missing} in '{directory}'. "
            f"Train the model first by running:\n    {TRAIN_COMMAND}\n"
            "This writes model.joblib and metadata.json into that directory."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return LoadedModel(pipeline=pipeline, metadata=metadata)
