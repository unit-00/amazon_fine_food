from pydantic import BaseModel
from typing import List, Tuple


class IdIn(BaseModel):
    pred_type: str
    iid: str

    class Config:
        schema_extra = {
            "example": {
                "pred_type": "item",
                "iid": "B000ET4SM8"
            }
        }

class PredictionOut(BaseModel):
    iid: str
    predictions: List[str]
