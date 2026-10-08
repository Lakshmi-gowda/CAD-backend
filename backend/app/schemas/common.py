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

class HealthData(BaseModel):
    status: str = "ok"
    model_loaded: bool = True

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
