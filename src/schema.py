from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    status: str = Field(..., description="PREDICT | REJECT")
    cataract_prob: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    action: str
