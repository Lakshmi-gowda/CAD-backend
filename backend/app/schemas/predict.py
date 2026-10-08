from typing import Any, Dict
from pydantic import BaseModel, Field

DEFAULT_DISCLAIMER = (
    "For decision support and educational purposes only. "
    "Not a substitute for formal diagnostic imaging."
)

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping exactly 52 feature names to their respective values",
    )

class PredictData(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: str
    disclaimer: str = DEFAULT_DISCLAIMER
