from typing import Any, Dict
from pydantic import BaseModel, Field

DEFAULT_DISCLAIMER = (
    "For decision support and educational purposes only. "
    "Not a substitute for formal diagnostic imaging. "
    "Artery model scores have not been clinically calibrated."
)

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Dictionary mapping the 56 saved artery-model feature names to their "
            "patient values. The existing overall CAD model uses its 52-feature subset."
        ),
    )

class ArteryPrediction(BaseModel):
    stenotic_probability: float
    predicted_class: str

class CadPrediction(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: str
    disclaimer: str = DEFAULT_DISCLAIMER

class PredictData(CadPrediction):
    artery_predictions: Dict[str, ArteryPrediction]
