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
        # PRD Rule 6: Probabilities must come directly from predict_proba()
        raw_proba = cad_model.predict_proba(df)
        if len(raw_proba) == 0:
            raise ValueError("predict_proba returned empty result")

        proba = raw_proba[0]
        # Target mapping: Normal: 0, CAD: 1
        prob_normal = float(proba[0])
        prob_cad = float(proba[1])

        probabilities: Dict[str, float] = {
            "CAD": prob_cad,
            "Normal": prob_normal,
        }

        # Predict label based on max probability / threshold
        prediction = "CAD" if prob_cad >= prob_normal else "Normal"

        return CadPrediction(
            prediction=prediction,
            probabilities=probabilities,
            model_version=registry.model_version,
            disclaimer=DEFAULT_DISCLAIMER,
        )

    except Exception as exc:
        logger.exception("Inference error during predict_cad: %s", exc)
        raise InternalError("Model inference failed")


def _class_value_for_label(
    label: str, classes: List[Any], metadata: Dict[str, Any]
) -> Any:
    if label in classes:
        return label

    class_mapping = metadata.get("class_mapping", {})
    if isinstance(class_mapping, dict) and label in class_mapping:
        value = class_mapping[label]
        if value in classes:
            return value

    if set(classes) == {0, 1}:
        if label == metadata.get("negative_class") == "Normal":
            return 0
        if label == metadata.get("positive_class") == "Stenotic":
            return 1

    raise ValueError(
        f"Cannot map model class {label!r} to one of classes_={classes!r}"
    )


def _label_for_class_value(value: Any, labels_and_values: Dict[str, Any]) -> str:
    matching_labels = [
        label
        for label, class_value in labels_and_values.items()
        if value == class_value
    ]
    if len(matching_labels) != 1:
        raise ValueError(f"Model returned an unmapped class value: {value!r}")
    return matching_labels[0]


def predict_arteries(
    df: pd.DataFrame, registry: ArtifactRegistry
) -> Dict[str, ArteryPrediction]:
    """Run each saved artery pipeline and map classes using artifact metadata."""
    predictions: Dict[str, ArteryPrediction] = {}
    metadata = registry.artery_metadata

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
            label_values = {
                label: _class_value_for_label(label, classes, metadata)
                for label in ("Normal", "Stenotic")
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

            stenotic_class_value = label_values["Stenotic"]
            stenotic_index = next(
                index
                for index, value in enumerate(classes)
                if value == stenotic_class_value
            )
            predicted_index = max(
                range(len(probabilities)), key=probabilities.__getitem__
            )
            predicted_class = _label_for_class_value(
                classes[predicted_index], label_values
            )
            predictions[artery] = ArteryPrediction(
                stenotic_probability=probabilities[stenotic_index],
                predicted_class=predicted_class,
            )
        except Exception as exc:
            logger.exception("%s artery inference failed: %s", artery, exc)
            raise InternalError(
                f"{artery} artery model inference failed",
                fields={"model": [artery]},
            ) from exc

    return predictions
