from fastapi import APIRouter

from backend.app.core.errors import ModelUnavailableError
from backend.app.schemas.common import SuccessResponse
from backend.app.schemas.predict import PredictRequest, PredictData
from backend.app.services.artifact_loader import get_artifact_registry
from backend.app.services.preprocessing import (
    validate_and_preprocess,
    validate_and_preprocess_artery_inputs,
)
from backend.app.services.model_service import predict_arteries, predict_cad

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=SuccessResponse[PredictData])
def predict(request: PredictRequest):
    registry = get_artifact_registry()
    if not registry.is_loaded:
        raise ModelUnavailableError("Model artifacts are not loaded")

    # Do not return a CAD-only success if an artery model is unavailable.
    for artery in ("LAD", "LCX", "RCA"):
        status = registry.artery_model_status.get(artery, {})
        if not status.get("loaded"):
            error = status.get("error") or "model is not loaded"
            raise ModelUnavailableError(
                f"{artery} artery model is unavailable: {error}",
                fields={"model": [artery]},
            )

    artery_features = validate_and_preprocess_artery_inputs(
        request.features, registry
    )
    cad_features = validate_and_preprocess(
        {
            feature: request.features[feature]
            for feature in registry.feature_names
            if feature in request.features
        },
        registry,
    )

    cad_result = predict_cad(cad_features, registry)
    artery_predictions = predict_arteries(artery_features, registry)
    result = PredictData(
        **cad_result.model_dump(),
        artery_predictions=artery_predictions,
    )

    return SuccessResponse(data=result)
