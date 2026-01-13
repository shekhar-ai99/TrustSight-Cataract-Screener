from pydantic import BaseModel, confloat
from typing import Literal


class InferenceOutputSchema(BaseModel):
    # Strict schema required by Phase 2
    cataract_prob: confloat(ge=0.0, le=1.0)
    confidence: confloat(ge=0.0, le=1.0)
    action: Literal["PREDICT", "REFER", "REJECT"]

    class Config:
        extra = "forbid"
