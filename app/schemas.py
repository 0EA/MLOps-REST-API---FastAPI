"""Pydantic v2 request/response models for the Iris prediction API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IrisMeasurementRequest(BaseModel):
    """Four flower measurements, in centimeters, in this exact field order."""

    model_config = ConfigDict(extra="forbid")

    sepal_length_cm: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Sepal length in centimeters. Must be a finite positive number.",
        examples=[5.1],
    )
    sepal_width_cm: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Sepal width in centimeters. Must be a finite positive number.",
        examples=[3.5],
    )
    petal_length_cm: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Petal length in centimeters. Must be a finite positive number.",
        examples=[1.4],
    )
    petal_width_cm: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Petal width in centimeters. Must be a finite positive number.",
        examples=[0.2],
    )


class IrisPredictionResponse(BaseModel):
    """Predicted species with the full class probability distribution."""

    predicted_species: str = Field(description="Name of the predicted Iris species.")
    probabilities: dict[str, float] = Field(
        description="Predicted probability for every species the model knows, keyed by species name."
    )
    model_version: str = Field(description="Version identifier of the model that produced this prediction.")


class HealthResponse(BaseModel):
    status: str = Field(description="Liveness status of the API process.")
    model_loaded: bool = Field(description="Whether the trained pipeline is loaded and ready to serve predictions.")


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded trained model."""

    model_version: str
    dataset: dict
    feature_order: list[str]
    feature_units: str
    target_classes: list[str]
    held_out_accuracy: float
    random_seed: int
    test_size: float
    package_versions: dict
    trained_at_utc: str
