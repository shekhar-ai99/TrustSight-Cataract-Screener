from typing import Literal, Optional
from pydantic import BaseModel, confloat


class InferenceOutput(BaseModel):
    status: Literal["PREDICT", "REJECT"]
    cataract_prob: Optional[confloat(ge=0.0, le=1.0)] = None
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    action: Literal["PREDICT", "REFER_TO_SPECIALIST", "REJECT"]
    reason: Optional[dict] = None

    class Config:
        extra = "forbid"
