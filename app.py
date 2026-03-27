from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re

app = FastAPI(
    title="AI Financial Assistant",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


class CategoryRequest(BaseModel):
    description: str


class CashflowRequest(BaseModel):
    month: int
    day: int
    weekday: int


class AnomalyRequest(BaseModel):
    amount: float


@app.get("/")
def home():
    return {"message": "API running 🚀"}


@app.post("/predict-category")
def predict_category(req: CategoryRequest):
    cat_model = joblib.load(os.path.join(MODEL_PATH, "categorization_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))

    text = clean_text(req.description)
    vec = vectorizer.transform([text])

    return {"category": cat_model.predict(vec)[0]}


@app.post("/predict-cashflow")
def predict_cashflow(req: CashflowRequest):
    cash_model = joblib.load(os.path.join(MODEL_PATH, "cashflow_model.pkl"))

    prediction = cash_model.predict([[req.month, req.day, req.weekday]])[0]

    return {"predicted_balance": float(prediction)}


@app.post("/detect-anomaly")
def detect_anomaly(req: AnomalyRequest):
    anomaly_stats = joblib.load(os.path.join(MODEL_PATH, "anomaly_stats.pkl"))

    mean = anomaly_stats['mean']
    std = anomaly_stats['std']

    z = (req.amount - mean) / std

    return {"anomaly": 1 if abs(z) > 3 else 0}