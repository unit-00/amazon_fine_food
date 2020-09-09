from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from model.model import predict

# Validation models
from backend.model.models import IdIn, PredictionOut


app = FastAPI()

@app.get('/')
async def root():
    return RedirectResponse('/docs')

@app.get('/hello')
def hello_world():
    return {'hello': 'world'}

@app.post('/predict', response_model=PredictionOut)
async def get_prediction(payload: IdIn):
    
    predictions = predict(payload.pred_type, payload.iid, 5)

    return {'iid': payload.iid, 'predictions': predictions}
