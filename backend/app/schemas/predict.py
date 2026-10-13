from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DISCLAIMER = (
    "For decision support and educational purposes only. "
    "Not a substitute for formal diagnostic imaging. "
    "Artery model scores have not been clinically calibrated."
)

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cad_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Values for the original CAD model's separate 52-feature contract.",
    )
    artery_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Values for the saved LAD, LCX, and RCA pipeline's 55-feature contract. "
            "Cath and LAD/LCX/RCA targets are excluded."
        ),
    )

class ArteryPrediction(BaseModel):
    stenotic_probability: float
    predicted_class: str
    probabilities: Dict[str, float]

class CadPrediction(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: str
    disclaimer: str = DEFAULT_DISCLAIMER

class PredictData(BaseModel):
    cad_prediction: Optional[CadPrediction] = None
    artery_predictions: Optional[Dict[str, ArteryPrediction]] = None
