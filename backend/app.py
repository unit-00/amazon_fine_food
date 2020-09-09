from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.model import predict

from typing import List, Tuple

app = FastAPI()

class IdIn(BaseModel):
    pred_type: str
    iid: str

class PredictionOut(BaseModel):
    iid: str
    predictions: List[str]

@app.get('/ping')
def pong():
    return {'ping': 'pong'}

@app.post('/predict', response_model=PredictionOut)
async def get_item_prediction(payload: IdIn):
    
    predictions = predict(payload.pred_type, payload.iid, 5)

    return {'iid': payload.iid, 'predictions': predictions}
