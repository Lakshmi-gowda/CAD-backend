from typing import List, Optional
from fastapi import APIRouter

from backend.app.core.errors import ModelUnavailableError
from backend.app.schemas.common import SuccessResponse, ModelInfoData, FeatureInfo
from backend.app.services.artifact_loader import get_artifact_registry

router = APIRouter(tags=["Model Info"])

@router.get("/model-info", response_model=SuccessResponse[ModelInfoData])
def get_model_info():
    registry = get_artifact_registry()
    if not registry.is_loaded:
        raise ModelUnavailableError("Model artifacts are not loaded")

    features_info: List[FeatureInfo] = []
    for feat_name in registry.feature_names:
        if feat_name in registry.encoders:
            encoder = registry.encoders[feat_name]
            allowed = list(getattr(encoder, "classes_", []))
            features_info.append(
                FeatureInfo(
                    name=feat_name,
                    type="categorical",
                    allowed_values=allowed,
                    required=True,
                )
            )
        else:
            features_info.append(
                FeatureInfo(
                    name=feat_name,
                    type="numeric",
                    allowed_values=None,
                    required=True,
                )
            )

    return SuccessResponse(
        data=ModelInfoData(
            model_name="XGBoost CAD classifier",
            model_version=registry.model_version,
            feature_count=len(registry.feature_names),
            features=features_info,
            classes=["Normal", "CAD"],
        )
    )
