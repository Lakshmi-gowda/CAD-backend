from fastapi import APIRouter
from backend.app.core.errors import ModelUnavailableError
from backend.app.schemas.common import SuccessResponse, HealthData
from backend.app.services.artifact_loader import get_artifact_registry

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=SuccessResponse[HealthData])
def health_check():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        raise ModelUnavailableError("Model artifacts are not loaded")

    return SuccessResponse(
        data=HealthData(status="ok", model_loaded=True)
    )
