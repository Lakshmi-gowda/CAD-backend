from fastapi import APIRouter

from backend.app.core.errors import ModelUnavailableError, ValidationError
from backend.app.schemas.common import SuccessResponse
from backend.app.schemas.predict import PredictRequest, PredictData
from backend.app.services.artifact_loader import get_artifact_registry
from backend.app.services.preprocessing import (
    validate_and_preprocess,
    validate_and_preprocess_artery_inputs,
)
from backend.app.services.model_service import predict_arteries, predict_cad

router = APIRouter(tags=["Prediction"])

@router.post(
    "/predict",
    response_model=SuccessResponse[PredictData],
    response_model_exclude_none=True,
)
def predict(request: PredictRequest):
    registry = get_artifact_registry()
    if request.cad_features is None and request.artery_features is None:
        raise ValidationError(
            "Provide cad_features, artery_features, or both",
            fields={"request": ["At least one prediction input set is required"]},
        )

    cad_result = None
    if request.cad_features is not None:
        if not registry.is_loaded:
            raise ModelUnavailableError(
                f"CAD model is unavailable: "
                f"{registry.load_error or 'artifacts not initialized'}",
                fields={"model": ["CAD"]},
            )
        cad_features = validate_and_preprocess(request.cad_features, registry)
        cad_result = predict_cad(cad_features, registry)

    artery_predictions = None
    if request.artery_features is not None:
        for artery in ("LAD", "LCX", "RCA"):
            status = registry.artery_model_status.get(artery, {})
            if not status.get("loaded"):
                error = status.get("error") or "model is not loaded"
                raise ModelUnavailableError(
                    f"{artery} artery model is unavailable: {error}",
                    fields={"model": [artery]},
                )
        artery_features = validate_and_preprocess_artery_inputs(
            request.artery_features, registry
        )
        artery_predictions = predict_arteries(artery_features, registry)

    return SuccessResponse(
        data=PredictData(
            cad_prediction=cad_result,
            artery_predictions=artery_predictions,
        )
    )
