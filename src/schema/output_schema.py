from pydantic import BaseModel, confloat
from typing import Literal


class InferenceOutputSchema(BaseModel):
    # Strict schema for federated evaluation
    prediction: Literal["CATARACT_PRESENT", "NORMAL"]
    confidence: confloat(ge=0.0, le=1.0)
    uncertainty: Literal["LOW", "MEDIUM", "HIGH"]
    action: Literal["PREDICT", "REFER_TO_SPECIALIST", "REJECT"]

    class Config:
        extra = "forbid"
