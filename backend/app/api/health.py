from fastapi import APIRouter
from backend.app.core.errors import ModelUnavailableError
from backend.app.schemas.common import SuccessResponse, HealthData, LivenessData
from backend.app.services.artifact_loader import get_artifact_registry

router = APIRouter(tags=["Health"])

@router.get("/health/live", response_model=SuccessResponse[LivenessData])
def liveness_check():
    return SuccessResponse(data=LivenessData(status="ok"))

@router.get("/health", response_model=SuccessResponse[HealthData])
def health_check():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        statuses = registry.get_model_status()
        raise ModelUnavailableError(
            "Overall CAD model is not loaded",
            fields={
                "model_status": [
                    f"{name}: loaded={status['loaded']}; error={status['error']}"
                    for name, status in statuses.items()
                ]
            },
        )

    return SuccessResponse(
        data=HealthData(
            status=(
                "ok"
                if all(
                    status["loaded"]
                    for status in registry.get_model_status().values()
                )
                else "degraded"
            ),
            model_loaded=True,
            model_status=registry.get_model_status(),
        )
    )
