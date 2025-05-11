from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Classifier API is up!"}

@app.post("/predict")
def predict_sentiment(item: TextInput):
    vec = joblib.load("model/vectorizer.joblib")
    model = joblib.load("model/model.joblib")
    X = vec.transform([item.text])
    pred = model.predict(X)[0]
    return {
        "prediction": "positive" if pred == 1 else "negative",
        "label": int(pred)
    }
