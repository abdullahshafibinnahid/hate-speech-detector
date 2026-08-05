"""
app.py - FastAPI for Gali Detector
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI(title="Multilingual Gali Detector API", version="1.0")

class TextInput(BaseModel):
    text: str

# Load TF-IDF model
model = None
vectorizer = None
if os.path.exists("gali_detector_model.pkl"):
    with open("gali_detector_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

@app.get("/")
def home():
    return {"status": "API running", "model_loaded": model is not None, "docs": "/docs"}

@app.post("/predict")
def predict(inp: TextInput):
    if model is None:
        return {"error": "Model not found. Train logistic_model.py first"}
    X = vectorizer.transform([inp.text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return {
        "text": inp.text,
        "prediction": "HATE" if int(pred)==1 else "NOT_HATE",
        "label": int(pred),
        "confidence": float(prob)
    }
