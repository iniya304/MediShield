from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    reliability_score: float
    q_score: float
    faithfulness: float
    robustness: float
    consistency: float
    decision: str
    gradcam_base64: str
    clinical_summary: Optional[str] = None
