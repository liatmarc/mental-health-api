from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("mental_health_model.pkl")
encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

class InputFeatures(BaseModel):
    avg_hours_streaming: float
    binge_sessions: int
    late_night_ratio: float
    pref_comedy: float
    pref_thriller: float
    sedentary_days: int

@app.post("/predict")
def predict(features: InputFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"mental_health_risk": encoder.inverse_transform(prediction)[0]}
