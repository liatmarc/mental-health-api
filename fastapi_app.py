from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and encoder
model = joblib.load("mental_health_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Define the input data schema
class InputData(BaseModel):
    avg_hours_streaming: float
    binge_sessions: int
    late_night_ratio: float
    pref_comedy: float
    pref_thriller: float
    sedentary_days: int

# Root route
@app.get("/")
def read_root():
    return {"message": "Mental Health Risk API is running. Visit /docs"}

# Prediction route
@app.post("/predict")
def predict(input_data: InputData):
    data = [[
        input_data.avg_hours_streaming,
        input_data.binge_sessions,
        input_data.late_night_ratio,
        input_data.pref_comedy,
        input_data.pref_thriller,
        input_data.sedentary_days
    ]]
    prediction = model.predict(data)
    result = encoder.inverse_transform(prediction)[0]
    return {"mental_health_risk": result}