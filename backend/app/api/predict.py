from fastapi import APIRouter

from backend.app.core.errors import ModelUnavailableError
from backend.app.schemas.common import SuccessResponse
from backend.app.schemas.predict import PredictRequest, PredictData
from backend.app.services.artifact_loader import get_artifact_registry
from backend.app.services.preprocessing import validate_and_preprocess
from backend.app.services.model_service import predict_cad

router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=SuccessResponse[PredictData])
def predict(request: PredictRequest):
    registry = get_artifact_registry()
    if not registry.is_loaded:
        raise ModelUnavailableError("Model artifacts are not loaded")

    # 1. Validate and preprocess features
    df = validate_and_preprocess(request.features, registry)

    # 2. Run inference
    result = predict_cad(df, registry)

    return SuccessResponse(data=result)
