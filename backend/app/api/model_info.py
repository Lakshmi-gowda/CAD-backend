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

    artery_features: List[FeatureInfo] = []
    if registry.artery_categorical_features:
        for feat_name in registry.artery_feature_names:
            if feat_name in registry.artery_categorical_features:
                artery_features.append(
                    FeatureInfo(
                        name=feat_name,
                        type="categorical",
                        allowed_values=registry.artery_category_values[feat_name],
                        required=True,
                    )
                )
            else:
                artery_features.append(
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
            artery_feature_count=len(registry.artery_feature_names),
            artery_features=artery_features,
            model_status=registry.get_model_status(),
        )
    )
