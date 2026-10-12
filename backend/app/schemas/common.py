from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")

class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: Optional[Dict[str, List[str]]] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail

class SuccessResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T

class ModelLoadStatus(BaseModel):
    loaded: bool
    error: Optional[str] = None
    feature_count: Optional[int] = None
    classes: List[str] = Field(default_factory=list)

class HealthData(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    model_status: Dict[str, ModelLoadStatus] = Field(default_factory=dict)

class LivenessData(BaseModel):
    status: str = "ok"

class FeatureInfo(BaseModel):
    name: str
    type: str  # "categorical" | "numeric"
    allowed_values: Optional[List[str]] = None
    required: bool = True

class ModelInfoData(BaseModel):
    model_name: str
    model_version: str
    feature_count: int
    features: List[FeatureInfo]
    classes: List[str]
    artery_feature_count: int = 0
    artery_features: List[FeatureInfo] = Field(default_factory=list)
    model_status: Dict[str, ModelLoadStatus] = Field(default_factory=dict)
