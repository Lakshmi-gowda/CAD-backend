import logging
import math
from typing import Any, Dict, List
import pandas as pd

from backend.app.core.errors import ModelUnavailableError, InternalError
from backend.app.schemas.predict import (
    ArteryPrediction,
    CadPrediction,
    DEFAULT_DISCLAIMER,
)
from backend.app.services.artifact_loader import ArtifactRegistry

logger = logging.getLogger("cad_backend")

def predict_cad(df: pd.DataFrame, registry: ArtifactRegistry) -> CadPrediction:
    """
    Executes inference using the registered XGBoost model on preprocessed features.
    Probabilities are sourced directly from predict_proba() without post-processing.
    """
    if not registry.is_loaded:
        raise ModelUnavailableError(
            f"Model is not loaded: {registry.load_error or 'artifacts not initialized'}"
        )

    cad_model = registry.models.get("cad")
    if cad_model is None:
        raise ModelUnavailableError("CAD model is not available in registry")

    try:
        classes = list(cad_model.classes_)
        raw_proba = cad_model.predict_proba(df)
        if len(raw_proba) != 1:
            raise ValueError(
                f"predict_proba returned {len(raw_proba)} rows; expected one"
            )
        proba = [float(value) for value in raw_proba[0]]
        if len(proba) != len(classes):
            raise ValueError("predict_proba output length does not match model.classes_")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in proba
        ):
            raise ValueError("predict_proba returned an invalid probability")

        class_labels = {
            value: label
            for label, value in registry.target_mapping.items()
        }
        probabilities = {
            class_labels[value]: probability
            for value, probability in zip(classes, proba)
            if value in class_labels
        }
        if len(probabilities) != len(classes):
            raise ValueError(
                f"Model classes_ cannot be mapped using target_mapping: {classes}"
            )

        raw_prediction = cad_model.predict(df)
        if len(raw_prediction) != 1 or raw_prediction[0] not in class_labels:
            raise ValueError("predict returned an unmapped CAD class")
        prediction = class_labels[raw_prediction[0]]

        return CadPrediction(
            prediction=prediction,
            probabilities=probabilities,
            model_version=registry.model_version,
            disclaimer=DEFAULT_DISCLAIMER,
        )

    except Exception as exc:
        logger.exception("Inference error during predict_cad: %s", exc)
        raise InternalError("Model inference failed")


def _class_value_for_label(label: str, classes: List[Any]) -> Any:
    if label in classes:
        return label

    raise ValueError(
        f"Cannot map model class {label!r} to one of classes_={classes!r}"
    )


def predict_arteries(
    df: pd.DataFrame, registry: ArtifactRegistry
) -> Dict[str, ArteryPrediction]:
    """Run each fitted artery pipeline and use its actual class ordering."""
    predictions: Dict[str, ArteryPrediction] = {}
    forbidden_columns = {"Cath", "LAD", "LCX", "RCA"} & set(df.columns)
    if forbidden_columns:
        raise InternalError(
            "Excluded features reached artery model inference",
            fields={"features": sorted(forbidden_columns)},
        )

    for artery in ("LAD", "LCX", "RCA"):
        status = registry.artery_model_status.get(artery, {})
        model = registry.artery_models.get(artery)
        if not status.get("loaded") or model is None:
            error = status.get("error") or "model is not loaded"
            raise ModelUnavailableError(
                f"{artery} artery model is unavailable: {error}",
                fields={"model": [artery]},
            )

        try:
            classes = list(model.classes_)
            class_labels = {
                value: _class_value_for_label(value, classes)
                for value in classes
            }
            raw_probabilities = model.predict_proba(df)
            if len(raw_probabilities) != 1:
                raise ValueError(
                    f"predict_proba returned {len(raw_probabilities)} rows; expected one"
                )
            probabilities = [float(value) for value in raw_probabilities[0]]
            if len(probabilities) != len(classes):
                raise ValueError(
                    "predict_proba output length does not match model.classes_"
                )
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in probabilities
            ):
                raise ValueError("predict_proba returned an invalid probability")

            raw_prediction = model.predict(df)
            if len(raw_prediction) != 1 or raw_prediction[0] not in classes:
                raise ValueError("predict returned an unmapped artery class")
            predicted_class = class_labels[raw_prediction[0]]
            named_probabilities = {
                class_labels[class_value]: probability
                for class_value, probability in zip(classes, probabilities)
            }
            predictions[artery] = ArteryPrediction(
                stenotic_probability=named_probabilities["Stenotic"],
                predicted_class=predicted_class,
                probabilities=named_probabilities,
            )
        except Exception as exc:
            logger.exception("%s artery inference failed: %s", artery, exc)
            raise InternalError(
                f"{artery} artery model inference failed",
                fields={"model": [artery]},
            ) from exc

    return predictions
