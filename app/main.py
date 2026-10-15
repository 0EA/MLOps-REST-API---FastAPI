"""FastAPI application serving the trained Iris classifier.

The trained scikit-learn pipeline and its metadata are loaded exactly once,
during application startup (see `lifespan` below), and reused for every
request. No training or reloading happens on the request path.
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from app.model import FEATURE_ORDER, load_model_artifacts
from app.schemas import (
    HealthResponse,
    IrisMeasurementRequest,
    IrisPredictionResponse,
    ModelInfoResponse,
)

# Resolved from this file's own location so the dashboard works regardless
# of the terminal's current working directory when uvicorn is started.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Raising here aborts startup. Uvicorn/FastAPI will print the exception
    # (including our actionable "run this command" message) and exit instead
    # of serving requests against a missing model.
    loaded = load_model_artifacts()
    app.state.pipeline = loaded.pipeline
    app.state.metadata = loaded.metadata
    yield
    app.state.pipeline = None
    app.state.metadata = None


app = FastAPI(
    title="Iris Classifier API",
    description="A small REST API serving a scikit-learn Iris classifier, built for an MLOps class demo.",
    version="1.0.0",
    lifespan=lifespan,
)


def _sanitize_for_json(value):
    """Replace non-finite floats with their string form.

    Starlette's JSONResponse serializes with allow_nan=False. A request that
    sends a literal `Infinity`/`NaN` token is correctly rejected by Pydantic
    (allow_inf_nan=False on our fields), but FastAPI's default validation
    error handler echoes the rejected value back in the response body, which
    would otherwise crash that serialization with an unrelated 500 instead of
    the intended 422.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _sanitize_for_json(exc.errors())})


# Static assets (CSS/JS) for the browser dashboard. Mounted under /static so
# it cannot shadow the API routes below, and served straight from disk --
# no bundler, no Node tooling.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the browser dashboard. Plain HTML/CSS/JS, no server-side templating."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    model_loaded = getattr(app.state, "pipeline", None) is not None
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["operations"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**app.state.metadata)


@app.post("/predict", response_model=IrisPredictionResponse, tags=["inference"])
def predict(request: IrisMeasurementRequest) -> IrisPredictionResponse:
    # Plain `def`, not `async def`: scikit-learn's predict/predict_proba are
    # blocking, CPU-bound calls. `async def` would not make this call
    # non-blocking or faster -- it would just run it directly on the event
    # loop and block every other request. FastAPI runs sync `def` endpoints
    # in a worker thread pool instead, which is the correct choice here. See
    # the README for a longer explanation.
    pipeline = app.state.pipeline
    metadata = app.state.metadata

    # Explicit feature vector in the exact order the model was trained on.
    feature_vector = np.array(
        [[getattr(request, name) for name in FEATURE_ORDER]],
        dtype=float,
    )

    predicted_label = pipeline.predict(feature_vector)[0]
    probabilities = pipeline.predict_proba(feature_vector)[0]

    # Map probabilities using the model's actual class order, not the order
    # they happen to appear in metadata.
    classes_ = pipeline.named_steps["classifier"].classes_
    target_classes = metadata["target_classes"]
    class_index_to_name = {cls: target_classes[i] for i, cls in enumerate(classes_)}

    probability_by_species = {
        class_index_to_name[cls]: float(prob) for cls, prob in zip(classes_, probabilities)
    }
    predicted_species = class_index_to_name[predicted_label]

    return IrisPredictionResponse(
        predicted_species=predicted_species,
        probabilities=probability_by_species,
        model_version=metadata["model_version"],
    )
