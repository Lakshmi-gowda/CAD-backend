import logging
from typing import Dict, Tuple, Any
import pandas as pd

from backend.app.core.errors import ModelUnavailableError, InternalError
from backend.app.schemas.predict import DEFAULT_DISCLAIMER, PredictData
from backend.app.services.artifact_loader import ArtifactRegistry

logger = logging.getLogger("cad_backend")

def predict_cad(df: pd.DataFrame, registry: ArtifactRegistry) -> PredictData:
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

        return PredictData(
            prediction=prediction,
            probabilities=probabilities,
            model_version=registry.model_version,
            disclaimer=DEFAULT_DISCLAIMER,
        )

    except Exception as exc:
        logger.exception("Inference error during predict_cad: %s", exc)
        raise InternalError("Model inference failed")
